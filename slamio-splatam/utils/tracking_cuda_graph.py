import torch


class TrackingStepGraph:
    """CUDA-graph capture for the tracking loop's optimizer.step() +
    zero_grad() pair.

    Captures only the Adam update on cam_unnorm_rots/cam_trans (the only
    two params left in the tracking optimizer after build_tracking_optimizer
    drops the zero-lr Gaussian-attribute groups) - everything else in a
    tracking iteration (render, loss, backward, early-stop) stays eager.
    The rasterizer's forward call does a synchronous cudaMemcpy internally
    (cuda_rasterizer/rasterizer_impl.cu, reading back the tile-instance
    count), which is illegal during CUDA graph stream capture, so
    render/backward can't be captured without patching the shared
    submodule - out of scope here.

    The tracking optimizer is reconstructed fresh every frame (new Adam
    state-tensor addresses), so a captured graph is only valid for the
    frame it was captured in. reset_for_frame() must be called once per
    frame, right after the new optimizer is built and before the first
    step() call on it.
    """

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", False))
        self.warmup_iters = int(cfg.get("warmup_iters", 3))

        self._graph = None
        self._warmup_done = 0
        self._capture_failed_this_frame = False
        # Across frames, unlike _capture_failed_this_frame which reset_for_frame
        # clears. A region that is not capture-safe fails on EVERY frame, and
        # each failed capture can leave the CUDA context damaged - retrying it
        # 592 times turned one bad capture into an out-of-memory crash. After
        # this many, give up and run eagerly for the rest of the run.
        self._capture_failures = 0
        self._max_capture_failures = int(cfg.get("max_capture_failures", 3))
        # Set when an outer graph (TrackingIterationGraph) has taken over and
        # disabled this one. The step then still runs eagerly here, but INSIDE
        # that outer capture - so grads must stay allocated. set_to_none=True
        # would free them, and the next backward would allocate fresh ones at
        # new addresses, which capture cannot tolerate.
        self.keep_grads = False

        self._frames_seen = 0
        self._frames_captured = 0
        self._frames_fallback = 0
        self._frames_warmup_only = 0
        self._total_replays = 0

        print(f"[TrackingStepGraph] constructed (enabled={self.enabled}, "
              f"warmup_iters={self.warmup_iters})", flush=True)

    def reset_for_frame(self):
        """Call once per frame, right after a new optimizer is built."""
        if self._frames_seen > 0:
            if self._graph is not None:
                self._frames_captured += 1
            elif self._capture_failed_this_frame:
                self._frames_fallback += 1
            else:
                self._frames_warmup_only += 1
        self._frames_seen += 1
        self._graph = None
        self._warmup_done = 0
        self._capture_failed_this_frame = False

    def step(self, optimizer=None, fn=None):
        """Replaces `optimizer.step(); optimizer.zero_grad(set_to_none=True)`.

        `fn` captures an arbitrary callable instead. The pose preconditioner
        REPLACES the optimizer update rather than feeding it (Adam renormalises
        per coordinate, so a pre-applied preconditioner is erased within a few
        iterations), so there is no optimizer.step() to record for it - but its
        update is just as capturable, provided the callable itself is
        branch-free and sync-free. See pose_preconditioner.se3_exp_capturable /
        mat_to_quat_capturable for the two helpers that had to be rewritten:
        both branched on a host-side float, which a capture would freeze at
        whichever branch it happened to record.
        """
        def _eager(set_to_none):
            if fn is not None:
                fn()
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=set_to_none)

        if not self.enabled or self._capture_failed_this_frame:
            _eager((not self.enabled) and not self.keep_grads)
            return

        if self._graph is not None:
            self._graph.replay()
            self._total_replays += 1
            return

        if self._warmup_done < self.warmup_iters:
            # Real eager step - lets Adam's lazily-allocated exp_avg/exp_avg_sq
            # state settle at a stable address before capture begins. The
            # preconditioner needs the warmup for a different reason: its P is
            # the identity until the first refactor, and that branch is decided
            # on the host, so capturing before it has run would freeze the
            # unpreconditioned path forever.
            _eager(False)
            self._warmup_done += 1
            return

        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _eager(False)
            # Capture only records the launches; run it once for real so
            # this iteration's actual gradient gets applied.
            graph.replay()
            self._graph = graph
            self._total_replays += 1
        except RuntimeError as e:
            # PERMANENT after repeated failures, not per-frame. A failed capture
            # can leave the CUDA context in a bad state, and _capture_failed_
            # this_frame is cleared by reset_for_frame() - so a genuinely
            # incompatible region was being retried on all 592 frames, turning
            # one bad capture into an out-of-memory crash further down the run.
            self._capture_failures += 1
            self._capture_failed_this_frame = True
            print(f"[TrackingStepGraph] capture failed "
                  f"({self._capture_failures}/{self._max_capture_failures}), "
                  f"falling back to eager: {e}", flush=True)
            if self._capture_failures >= self._max_capture_failures:
                self.enabled = False
                print("[TrackingStepGraph] DISABLED for the rest of the run - "
                      "the captured region is not capture-safe. The run "
                      "continues eagerly; timings are graph-off.", flush=True)
            _eager(False)

    def summary(self) -> str:
        if not self.enabled:
            return "Tracking-step CUDA graph capture: disabled"
        captured, fallback, warmup_only = (
            self._frames_captured, self._frames_fallback, self._frames_warmup_only,
        )
        if self._frames_seen > 0:
            if self._graph is not None:
                captured += 1
            elif self._capture_failed_this_frame:
                fallback += 1
            else:
                warmup_only += 1
        return (f"Tracking-step CUDA graph capture: {captured}/{self._frames_seen} frames "
                f"captured+replayed, {fallback} fell back to eager, {warmup_only} never "
                f"reached the warmup threshold, {self._total_replays} total graph replays")
