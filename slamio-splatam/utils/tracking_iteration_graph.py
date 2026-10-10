import contextlib
import gc
import os

import torch


@contextlib.contextmanager
def _gc_collect_suppressed():
    """Make gc.collect() a no-op for the duration, then restore it exactly.

    Restored in a finally, so an exception inside cannot leave the process
    with collection permanently disabled. Only the EXPLICIT call is skipped
    here; see _gc_pause for the automatic (threshold-triggered) collector,
    which has to be handled separately.
    """
    real = gc.collect
    gc.collect = lambda *a, **k: 0
    try:
        yield
    finally:
        gc.collect = real


def _gc_pause():
    """Disable AUTOMATIC garbage collection; return whether it was on.

    MEASURED FAILURE (MonoGS fr1_desk, frame ~300, ITERGRAPH_SKIP_GC=1):
    'CUDA error: operation not permitted when stream is capturing', raised
    from a tensor destructor reached through PyType_GenericAlloc - i.e. the
    cyclic collector ran INSIDE the capture and freed an old tensor, whose
    allocator bookkeeping issues a CUDA call the capture forbids. It is a
    hard abort (terminate called ... c10::Error), not a fallback to eager.

    torch.cuda.graph's own gc.collect() is what protected against this: it
    reset the collector's allocation counters just before the capture, so
    a threshold-triggered pass was unlikely to land inside it. Skipping
    the call removed that protection and left the counters running, so a
    pass now lands inside the capture with probability that grows with
    every frame. Two full runs got through on luck.

    Pausing the collector for the capture window is the fix, and is safer
    than stock: it makes a pass inside the capture impossible rather than
    unlikely. The cost is nothing - collection resumes right after and
    catches up. Process-wide, so it also covers other Python threads.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    return was_enabled


def _gc_resume(was_enabled):
    if was_enabled:
        gc.enable()


class _CaptureNoGC(torch.cuda.graph):
    """torch.cuda.graph whose __enter__ does not run gc.collect().

    MEASURED, NOT ASSUMED. Profiling the branch on Replica room0 (776K
    Gaussians, 100 frames) put 17.4 s in gc.collect - 178 ms per frame,
    ONE call per frame, from torch.cuda.graph.__enter__. It is a pure CPU
    call with no GPU wait inside it, so cProfile does not inflate it, and it
    scales with the number of live Python objects rather than with the
    tracking iteration it precedes. On that scene it costs about as much per
    run as everything the branch's tracking loop saves over stock main.

    WHAT IT WAS FOR: __enter__ synchronises, collects garbage and empties the
    cache so the capture starts from a clean allocator. Only the collection
    is skipped here; synchronize() and empty_cache() still run.

    WHAT COULD GO WRONG, and why it is opt-in: the documented capture failure
    on this branch (legacy stream depends on a capturing blocking stream) is
    a LIVE REFERENCE to the previous iteration keeping warmup-era autograd
    accumulators alive. That is fixed by dropping the references, which gc
    cannot do for a live object - but a reference CYCLE holding such a
    tensor is exactly what gc.collect would have freed. So a run with this
    on must show every frame captured and none falling back.
    """

    def __enter__(self):
        self._gc_was_enabled = _gc_pause()
        try:
            with _gc_collect_suppressed():
                return super().__enter__()
        except BaseException:
            _gc_resume(self._gc_was_enabled)
            raise

    def __exit__(self, *exc_info):
        try:
            return super().__exit__(*exc_info)
        finally:
            _gc_resume(self._gc_was_enabled)


class TrackingIterationGraph:
    """Captures a whole tracking iteration - render, loss, backward, optimizer
    step - into a CUDA graph and replays it.

    This is the payoff for the capture-safe work elsewhere. A tracking
    iteration issues roughly 230 kernel launches; a graph replay issues one.
    With the GPU measured only ~40% busy, the loop is CPU-dispatch-bound, so
    collapsing the launches is where the time is.

    Prerequisites, all satisfied before this class is useful:
      - static loss shapes            tracking.mask_multiply_loss
      - no rasterizer host readback   tracking.binning_capacity
      - fixed buffer addresses        follows from the fixed capacity
      - syncs after the captured region: pose_delta_norm.item(), the candidate
        comparison and stopper.check all run after optimizer.step(), so they
        do not interrupt the capture

    How values get out. Tensors produced inside a capture live in the graph's
    private memory pool, and replay rewrites that same memory. So references
    taken at capture time keep reading current values after each replay -
    that is how loss and losses reach the early-stop check without re-running
    any Python.

    What this does NOT handle: anything that changes shape or control flow
    within a frame. Gaussian count is fixed during tracking, and the tile mask
    is built once per frame, so the region is static by construction. Across
    frames both change, hence the per-frame recapture.
    """

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", False))
        self.warmup_iters = int(cfg.get("warmup_iters", 5))
        # "global" (PyTorch's default) errors on any legacy-default-stream
        # operation anywhere in the process during capture. Autograd runs
        # backward on worker threads and "current stream" is thread-local, so
        # a backward node can legitimately be on a different stream than the
        # capture - "thread_local" restricts the check to this thread and
        # allows that. "relaxed" is looser still.
        self.capture_error_mode = str(cfg.get("capture_error_mode", "global"))
        # OFF BY DEFAULT, AND IT SHOULD STAY OFF. Measured, 4-frame real run:
        #
        #   side_stream_warmup=True   illegal memory access at
        #                             pose_delta_norm.item(), frame 1
        #   side_stream_warmup=False  3/3 frames captured, 242 replays
        #
        # The idea was sound in isolation - PyTorch's capture recipe does want
        # warmup on a side stream, and it does remove any need to bridge streams
        # for a pinned AccumulateGrad. It passed as fixA in
        # profiling/repro_capture.py. But the reproducer ran NOTHING on the
        # default stream between warmup iterations, and the real loop runs
        # plenty: pose_delta_norm, the candidate comparison, stopper.check, and
        # BinningCapacity's _count_out.item() and _overflow.zero_().
        #
        # That is what breaks. iteration_fn() allocates on the side stream;
        # PyTorch's caching allocator tags each block with its allocating stream
        # and may hand a freed block straight back out on that stream, so a
        # tensor allocated on the side stream and then read from the default
        # stream needs record_stream() or it can be reused while still live. The
        # wait_stream fences below order the STREAMS and tell the ALLOCATOR
        # nothing.
        #
        # The blocker this was meant to help with is already fixed by dropping
        # the references that pinned the warmup graph (splatam.py, verified
        # independently as fixB). This flag adds nothing on top of that and
        # costs correctness. Kept only so the A/B can be repeated.
        self.side_stream_warmup = bool(cfg.get("side_stream_warmup", False))
        # skip_gc / ITERGRAPH_SKIP_GC=1: enter the capture without gc.collect().
        # Off by default. See _CaptureNoGC for the measurement behind it and
        # for what has to be checked before trusting it. The environment
        # variable overrides the config, and is read HERE rather than in each
        # model so SplaTAM, MonoGS and Gaussian-SLAM all honour it identically.
        self.skip_gc = bool(cfg.get("skip_gc", False))
        _env_gc = os.environ.get("ITERGRAPH_SKIP_GC")
        if _env_gc is not None:
            self.skip_gc = _env_gc not in ("0", "", "false", "False")

        self._graph = None
        self._outputs = None
        self._warmup_done = 0
        self._failed_this_frame = False
        # Which KIND of iteration the current graph was captured for. See run().
        self._phase = 0
        self.phase_switches = 0
        # Iterations handed back to the caller to run eagerly. Reported
        # so a run cannot silently bypass everything and still call
        # itself a graph run.
        self.bypassed = 0
        # One stream for the whole run: it only has to be the same object for
        # the warmup and the capture of a given frame, and reusing it avoids
        # allocating a stream per frame.
        self._side_stream = torch.cuda.Stream() if self.side_stream_warmup else None

        self.frames = 0
        self.frames_captured = 0
        self.frames_fallback = 0
        self.replays = 0
        self._first_error = None

        print(f"[TrackingIterationGraph] constructed (enabled={self.enabled}, "
              f"warmup_iters={self.warmup_iters}, "
              f"side_stream_warmup={self.side_stream_warmup}, "
              f"capture_error_mode={self.capture_error_mode}, "
              f"skip_gc={self.skip_gc})", flush=True)

    def _log(self, msg):
        """Flushed, first frame only. Enough to locate a fault without
        producing 592 frames of noise."""
        if self.frames <= 1:
            print(f"  [itergraph] {msg}", flush=True)

    def reset_for_frame(self):
        # Drop the previous frame's graph before the next capture. The
        # optimizer is rebuilt per frame and the Gaussian count may have
        # changed in mapping, so nothing from the last frame is reusable.
        self._graph = None
        self._outputs = None
        self._warmup_done = 0
        self._failed_this_frame = False
        self._phase = 0
        self.frames += 1

    def will_capture(self) -> bool:
        """True if the next non-bypassed run() would CAPTURE.

        Exists so a caller can drop live references to the previous
        iteration before the capture begins. A tensor still referenced
        from iteration N-1 keeps its warmup-era autograd accumulators
        alive, and the capture then reuses nodes stamped with the legacy
        default stream instead of rebuilding them on the capture stream -
        which fails as

            operation would make the legacy stream depend on a
            capturing blocking stream

        raised from inside backward(), not from the line holding the
        reference. SplaTAM hit this first and fixes it with an explicit
        `loss = losses = None` before the capture iteration; it is
        isolated in profiling/repro_capture.py, where rung 4d fails on the
        single addition of a live reference to the previous loss.

        Gradient reuse reintroduces exactly that hazard, because holding
        the previous iteration's outputs is the whole mechanism.
        """
        return (self.enabled and self._graph is None
                and not self._failed_this_frame
                and self._warmup_done >= self.warmup_iters)

    def run(self, iteration_fn, phase: int = 0, bypass: bool = False):
        """Run one tracking iteration, capturing or replaying as appropriate.

        iteration_fn() must perform render + loss + backward + optimizer step
        and return whatever the caller needs afterwards (loss, variables,
        losses). It must not synchronise, allocate host-visible state, or
        branch on device values.
        """
        if not self.enabled:
            return iteration_fn()

        # BYPASS: run THIS iteration eagerly and leave the graph alone.
        #
        # Gradient reuse makes the iteration body branch - some
        # iterations render, some restore a stashed gradient and skip
        # straight to the step - and a capture records ONE fixed kernel
        # sequence. The two were refused together for exactly that
        # reason: a graph captured on a rendering iteration would replay
        # on a reusing one and silently re-render it.
        #
        # A SECOND GRAPH IS THE WRONG FIX. cudaGraphInstantiate is ~57 ms
        # and pays only when spread over enough replays; capturing a
        # second one for the reuse body would pay it twice per frame and
        # split the replay count that has to amortise it. The reuse body
        # has no render in it, so there is very little there to capture
        # anyway - it is the cheap half of the iteration by construction.
        #
        # SO THE CHEAP HALF RUNS EAGER AND THE EXPENSIVE HALF REPLAYS.
        # This returns BEFORE the phase check, before the replay and
        # before the warmup counter, which is the whole correctness
        # argument: a bypassed iteration cannot advance _warmup_done,
        # cannot trigger or invalidate a capture, and cannot be mistaken
        # for a phase change. The capture therefore always lands on a
        # rendering iteration, by construction rather than by luck. A
        # phase change seen on a bypassed iteration is DEFERRED to the
        # next real one, not lost.
        if bypass:
            self.bypassed += 1
            return iteration_fn()

        # PHASE CHANGE => NEW CAPTURE. A captured graph replays exactly the
        # kernels it recorded, so a frame whose iterations are not all alike
        # needs one graph per KIND of iteration. The preconditioner's handoff
        # is exactly that: iterations 0..N-1 take the preconditioned update and
        # the rest take Adam's, at a fixed, known switch point.
        #
        # Dropping the graph here makes the new phase warm up and capture on
        # its own, so each phase replays its own kernels.
        #
        # IT IS NOT FREE, AND THE COST MODEL SAYS SO. cudaGraphInstantiate is
        # ~57 ms per capture and pays only when spread over enough replays:
        # ~195 replays/frame gave 2.20x on Gaussian-SLAM, ~74 gave -18% on
        # MonoGS. Two phases pays it TWICE per frame and runs the eager warmup
        # twice, over a REDUCED replay count - because cutting iterations is
        # the whole point of the preconditioner. Measure it; do not assume the
        # single-phase speedup carries over.
        if phase != self._phase:
            self._phase = phase
            self._graph = None
            self._outputs = None
            self._warmup_done = 0
            self._failed_this_frame = False
            self.phase_switches += 1

        if self._graph is not None:
            self._graph.replay()
            self.replays += 1
            return self._outputs

        if self._warmup_done < self.warmup_iters:
            # Eager iterations. These also let BinningCapacity measure
            # num_rendered before the capacity is frozen, and give the caching
            # allocator a chance to settle.
            self._warmup_done += 1
            # First frame only, flushed: a CUDA fault kills the process with
            # buffered stdout still unwritten, so without this there is no way
            # to tell an EAGER warmup failure from a capture failure - and that
            # distinction decided the whole reproducer investigation.
            self._log(f"warmup {self._warmup_done}/{self.warmup_iters} (eager"
                      f"{', side stream' if self._side_stream else ''})")
            if self._side_stream is None:
                return iteration_fn()
            # Fence in both directions so the warmup still observes, and is
            # observed by, everything the caller does on the default stream
            # between iterations - pose_delta_norm, the candidate comparison
            # and the early-stop check all run there.
            self._side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._side_stream):
                outputs = iteration_fn()
            torch.cuda.current_stream().wait_stream(self._side_stream)
            return outputs

        try:
            graph = torch.cuda.CUDAGraph()
            self._log("entering capture")
            _capture_ctx = _CaptureNoGC if self.skip_gc else torch.cuda.graph
            with _capture_ctx(graph, stream=self._side_stream,
                              capture_error_mode=self.capture_error_mode):
                outputs = iteration_fn()
            self._log("capture returned, replaying")
            # Capture records the work without performing it, so replay once
            # to actually apply this iteration.
            graph.replay()
            self._log("replay done - frame captured")
            self._graph = graph
            self._outputs = outputs
            self.replays += 1
            self.frames_captured += 1
            return outputs
        except RuntimeError as e:
            # Deliberately fatal. Falling back to eager after a failed capture
            # was tried and is NOT safe: a capture that aborts partway leaves
            # the CUDA context and caching allocator inconsistent, and the run
            # continues producing garbage - the first attempt carried on to
            # frame 1 and died in torch.inverse() on a singular pose matrix,
            # far from the real cause. A hard stop at the point of failure is
            # more useful than a corrupted run.
            self._first_error = str(e)
            self.frames_fallback += 1
            raise RuntimeError(f"""CUDA graph capture of the tracking iteration failed.
This is not recoverable in-process - the context is left inconsistent, so the
run is aborted rather than continued.

  Re-run with CUDA_LAUNCH_BLOCKING=1 to get the real failing operation:
  capture errors surface asynchronously, so without it the traceback points
  at capture_end() rather than the offending call.

  Capture rejects anything with a data-dependent shape or a host sync -
  boolean-mask indexing or assignment, .item(), nonzero(), and kernels
  launched on the legacy default stream rather than PyTorch's current stream.

  Underlying error: {e}""") from e

    def summary(self) -> str:
        if not self.enabled:
            return "Tracking-iteration CUDA graph: disabled"
        s = (f"Tracking-iteration CUDA graph: {self.frames_captured}/{self.frames} "
             f"frames captured, {self.frames_fallback} fell back to eager, "
             f"{self.replays} replays")
        if self.skip_gc:
            s += " [capture entered without gc.collect]"
        if self.phase_switches:
            # Two captures per frame is a DIFFERENT cost model from one - say
            # so, rather than letting a halved replay count look like the same
            # configuration.
            s += (f"\n  {self.phase_switches} phase switches "
                  f"({self.phase_switches / max(self.frames, 1):.2f}/frame) - "
                  f"each forces an extra warmup and capture")
        if self.bypassed:
            # THE DENOMINATOR FOR THE REPLAY COUNT. With gradient reuse
            # on, a bypassed iteration is one the caller ran eagerly on
            # purpose, so replays/frame is no longer replays/iterations -
            # and the graph's whole cost model is replays-per-capture.
            # Report both, so a run that bypassed half its iterations
            # cannot be read as a graph run with a disappointing replay
            # count.
            _tot = self.replays + self.bypassed
            s += (f"\n  {self.bypassed} iterations bypassed (eager, "
                  f"{100.0 * self.bypassed / max(_tot, 1):.1f}% of "
                  f"{_tot} graph-eligible), "
                  f"{self.replays / max(self.frames_captured, 1):.1f} "
                  f"replays per capture")
        if self._first_error is not None:
            s += f"\n  first capture error: {self._first_error}"
        return s
