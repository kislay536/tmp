"""Overlap frame loading with compute. Shared by SplaTAM, MonoGS and Gaussian-SLAM.

WHY THIS EXISTS. Every model in this framework loads its next frame
SYNCHRONOUSLY, inline in the frame loop, and then computes on it. MonoGS is the
one that has been measured: slam_frontend.py times the load explicitly and it
came out at 22-26s against a Total time of ~270s on epyc8 - 8-9% of the run
spent decoding JPEGs while the GPU idles. Nothing in the ladder had looked at
it, because every rung so far measured tracking or mapping ms/iter and this
sits outside both.

The access pattern is the easy case: strictly sequential, one frame at a time,
index known one step ahead. A worker thread reading dataset[i+1] while the main
thread computes on frame i removes the wait entirely, up to the point where
decode is slower than the frame's compute.

A THREAD IS THE RIGHT TOOL, not a process. The expensive work is PIL / cv2 /
imageio decode and numpy arithmetic, all of which RELEASE THE GIL, so a thread
genuinely overlaps. A process would duplicate the CUDA context and force the
samples through IPC, which is how you end up with the "Producer process has been
terminated before all shared CUDA tensors released" warnings this repo already
produces.

  ============================================================
  THE CONTRACT: THE WRAPPED __getitem__ MUST NOT TOUCH CUDA
  ============================================================

  This is not a style preference, it is a correctness requirement, and it is why
  this class does not simply thread the existing loaders as they are.

  CUDA graph capture runs with cudaStreamCaptureModeGlobal. In that mode ANY
  thread launching work into a non-capturing stream during a capture is an
  error. SplaTAM captures a tracking iteration on EVERY frame and Gaussian-SLAM
  does too, so a prefetch thread that issued an H2D copy at the wrong moment
  would break capture nondeterministically - a failure that would look like the
  capture bugs already documented in the ladder rather than like a dataloader
  bug.

  There is a second, quieter reason. The caching allocator tags each block with
  the stream it was allocated on and can hand a freed block straight back out on
  that stream. That is exactly what made side_stream_warmup produce an illegal
  memory access after passing its own reproducer. Keeping CUDA out of the worker
  removes the whole class of problem instead of reasoning about it.

  So: the worker produces CPU tensors / numpy arrays. The CONSUMER moves them to
  the device, on the main thread, where it always happened. MonoGS's dataset
  ordinarily returns CUDA tensors, so its integration constructs the dataset on
  CPU and moves in Camera.init_from_dataset - see that call site.

  A prefetcher is NOT a place to also fuse the H2D copy. Pinning and
  non_blocking=True are a separate, later change with their own measurement.

  ============================================================
  HOW TO TELL IT IS ACTUALLY WORKING
  ============================================================

  summary() prints hits, misses and - the number that matters - the total time
  the CONSUMER SPENT BLOCKED waiting for a frame that had not arrived yet. If
  hits are high and wait is near zero, loading is fully hidden and the speedup
  is real. If wait is close to the old load time, the worker is not keeping up
  and the win is smaller than the load figure suggested.

  Read that before believing any timing difference. A prefetcher that misses
  every frame still "works" and buys nothing, and it would be invisible in a
  wall-clock number.
"""

import collections
import queue
import threading
import time


class _WorkerError:
    """Carries a worker-thread exception across the queue to the consumer.

    Re-raised on the main thread rather than logged, because a dataset that
    fails to load must not silently become a skipped frame - it would show up
    much later as an inexplicable trajectory rather than as an IOError.
    """

    def __init__(self, exc):
        self.exc = exc


class FramePrefetcher:
    """Wraps any indexable dataset and reads ahead on a worker thread.

    Semantics are exactly those of the wrapped object: prefetched[i] returns
    what dataset[i] returns. Non-sequential access is supported and simply
    costs a resync, so a model that reads dataset[0] at start-up and then walks
    forward - which all three do - pays one resync and nothing else.
    """

    def __init__(self, dataset, cfg: dict = None, name="dataset", to_device=None):
        """to_device: optional callable(sample) -> sample, applied ON THE
        CONSUMER THREAD to every sample, hit or miss alike.

        This is how a model whose __getitem__ ordinarily returns CUDA tensors
        gets wrapped WITHOUT editing its call sites. Construct the dataset on
        CPU so the worker stays CUDA-free, and hand the host-to-device move in
        here: it then runs on whatever thread asked for the frame, which is
        where it ran before. SplaTAM reads its datasets from seven places, so
        doing this at the call sites instead would mean seven chances to miss
        one - and a missed one is a CPU tensor reaching the rasterizer.

        MonoGS does the same move explicitly inside Camera.init_from_dataset
        because it has exactly one call site and the Camera constructor needs
        the device anyway. Both routes satisfy the same contract.
        """
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self._to_device = to_device

        # Queue depth. 2-3 is plenty: the consumer needs ONE frame ready, and
        # anything deeper only buys tolerance for a jittery decode time while
        # costing that many decoded frames of host memory (a TUM RGB frame is
        # ~1.8 MB as float32, so depth 3 is ~5 MB - negligible, but there is no
        # reason to go deeper without evidence of jitter).
        #
        # Named queue_depth, not depth: this object impersonates a dataset via
        # __getattr__, and `depth` is what a dataset's own depth image would be
        # called. A collision there would resolve silently to the wrong thing.
        self.queue_depth = max(1, int(cfg.get("depth", 3)))

        # How many already-returned samples to keep. 0 disables the cache.
        #
        # Needed by the SERIAL models, not by MonoGS. Gaussian-SLAM's tracker
        # reads dataset[frame_id] and dataset[frame_id - 1]; SplaTAM reads
        # dataset[time_idx] from two places in a frame. Against a pure forward
        # queue each of those tears down the worker and rebuilds it, so the hit
        # rate collapses and the thread churn costs more than the prefetch
        # saves. Two slots covers both patterns.
        #
        # ALIASING: a cache hit returns THE SAME OBJECT as the first read, not
        # a fresh load. Every consumer in this repo derives new tensors
        # (permute, /255, .to()) rather than mutating in place, which is why
        # this is safe here - but a caller that mutated a sample would corrupt
        # the next read of the same index. That is why it defaults to 0 and is
        # turned on per model rather than assumed.
        self.recent_size = max(0, int(cfg.get("recent", 0)))
        self._recent = collections.OrderedDict()
        self._cache_hits = 0

        self._ds = dataset
        self._name = name

        self._q = None
        self._thread = None
        self._stop = None
        self._start_idx = 0      # index the current worker began at
        self._taken = 0          # items the consumer has pulled from this worker

        # Instrumentation. See the docstring: `wait` is the load-bearing one.
        self._hits = 0
        self._misses = 0
        self._resyncs = 0
        self._wait_s = 0.0
        self._direct_s = 0.0

    # -- delegation -------------------------------------------------------
    # Models read dataset.fx, dataset.height, dataset.poses and friends all
    # over the place. Forward anything we do not define ourselves so the
    # wrapper is a drop-in replacement rather than something every call site
    # has to be taught about.

    def __getattr__(self, item):
        # Only called when normal lookup fails, so it cannot shadow our own
        # attributes. _ds is set in __init__ before anything can reach here.
        return getattr(object.__getattribute__(self, "_ds"), item)

    def __len__(self):
        return len(self._ds)

    @property
    def dataset(self):
        """The wrapped object, for code that needs the real thing."""
        return self._ds

    # -- the worker -------------------------------------------------------

    def _run(self, start, stop_event, q):
        i = start
        n = len(self._ds)
        while not stop_event.is_set() and i < n:
            try:
                sample = self._ds[i]
            except Exception as exc:  # noqa: BLE001 - re-raised on the consumer
                sample = _WorkerError(exc)
            # Bounded put with a timeout so a stopped worker cannot wedge on a
            # full queue that nobody will ever drain. Without the timeout,
            # close() would block until the consumer happened to take one more.
            while not stop_event.is_set():
                try:
                    q.put((i, sample), timeout=0.1)
                    break
                except queue.Full:
                    continue
            else:
                return
            if isinstance(sample, _WorkerError):
                return
            i += 1

    def _start_worker(self, idx):
        self._stop_worker()
        self._q = queue.Queue(maxsize=self.queue_depth)
        self._stop = threading.Event()
        self._start_idx = idx
        self._taken = 0
        self._thread = threading.Thread(
            target=self._run,
            args=(idx, self._stop, self._q),
            name=f"prefetch-{self._name}",
            daemon=True,   # never hold up interpreter shutdown
        )
        self._thread.start()

    def _stop_worker(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        self._q = None
        self._stop = None

    # -- access -----------------------------------------------------------

    def _take(self):
        t0 = time.perf_counter()
        idx, sample = self._q.get()
        self._wait_s += time.perf_counter() - t0
        self._taken += 1
        if isinstance(sample, _WorkerError):
            raise sample.exc
        return idx, sample

    def _deliver(self, idx, sample):
        """Consumer-thread post-processing, then remember the result.

        Every return path in __getitem__ goes through here, including the
        disabled one, so turning prefetch off cannot change what a call site
        receives. The host-to-device move lives here precisely because this
        runs on the CALLING thread - never on the worker.
        """
        if self._to_device is not None:
            sample = self._to_device(sample)
        if self.recent_size:
            self._recent[idx] = sample
            while len(self._recent) > self.recent_size:
                self._recent.popitem(last=False)
        return sample

    def __getitem__(self, idx):
        if not self.enabled:
            return self._deliver(idx, self._ds[idx])

        if idx < 0:
            idx += len(self._ds)

        # A REPEATED OR BACKWARD READ IS NOT A RESYNC IF IT IS STILL IN HAND.
        # Both serial models re-read within a frame - Gaussian-SLAM's tracker
        # takes frame_id and frame_id-1, SplaTAM reads dataset[time_idx] from
        # two places - and against a pure forward queue every one of those
        # would tear the worker down and rebuild it. The hit rate collapses and
        # the thread churn costs more than the prefetch saves.
        if self.recent_size and idx in self._recent:
            self._recent.move_to_end(idx)
            self._cache_hits += 1
            return self._recent[idx]

        if self._thread is None:
            self._start_worker(idx)

        # The worker emits strictly increasing indices from _start_idx, and the
        # consumer has taken _taken of them, so the next item the queue will
        # yield is known WITHOUT peeking - which queue.Queue cannot do anyway.
        head = self._start_idx + self._taken

        if idx == head:
            got, sample = self._take()
            self._hits += 1
            return self._deliver(idx, sample)

        # A small forward jump: discard what was prefetched and skipped rather
        # than tearing the worker down. Bounded by the queue depth, since past
        # that the worker cannot have run ahead far enough for draining to be
        # cheaper than restarting.
        if head < idx <= head + self.queue_depth:
            while self._start_idx + self._taken < idx:
                self._take()
            got, sample = self._take()
            self._hits += 1
            return self._deliver(idx, sample)

        # Anything else - a backward read, a big jump, or a worker that has run
        # off the end of the dataset - is a resync. Load this one directly and
        # aim the worker at the NEXT index, which is what a sequential walk will
        # ask for.
        self._misses += 1
        self._resyncs += 1
        t0 = time.perf_counter()
        sample = self._ds[idx]
        self._direct_s += time.perf_counter() - t0
        self._start_worker(idx + 1)
        return self._deliver(idx, sample)

    # -- lifecycle --------------------------------------------------------

    def close(self):
        self._stop_worker()
        # Drop cached samples too. Post-to_device they may be CUDA
        # tensors, and a process trying to exit should not be holding
        # device memory in a dataloader.
        self._recent.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter teardown, nothing to do
            pass

    # -- reporting --------------------------------------------------------

    def summary(self) -> str:
        if not self.enabled:
            return f"Frame prefetch ({self._name}): disabled"
        total = self._hits + self._misses
        if total == 0:
            return f"Frame prefetch ({self._name}): enabled but never used"
        return (
            f"Frame prefetch ({self._name}): {self._hits}/{total} hits "
            f"({100.0 * self._hits / total:.1f}%), {self._resyncs} resyncs, "
            f"queue_depth={self.queue_depth}, {self._cache_hits} cache hits "
            f"(recent={self.recent_size}), consumer blocked {self._wait_s:.1f}s, "
            f"direct loads {self._direct_s:.1f}s"
        )
