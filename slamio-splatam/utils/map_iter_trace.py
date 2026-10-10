"""Per-iteration MAPPING loss trace - the mapping-side counterpart to
utils/iter_trace.py, which is tracking-only.

Deliberately much lighter than IterTrace: mapping optimises millions of
Gaussian parameters (see the pose_preconditioner.py "why not mapping"
argument), so there is no cheap per-iteration "distance to a final answer" to
capture the way there is for the 6-D pose. What IS cheap and directly
answers "does mapping settle before its iteration budget" is the mapping
LOSS itself, one scalar per iteration - that is all this captures.

Turn on with MAP_ITER_TRACE=<path.jsonl> (or mapping.iter_trace.out_path).
MAP_ITER_TRACE_EVERY=N (or mapping.iter_trace.every) traces only frames with
frame % N == 0, same convention as ITER_TRACE_EVERY.

NOT A TIMED RUN: capture() does a host sync every mapping iteration
(loss.item()) exactly like the tracking-side ITER_TRACE already does.

One JSON line per traced frame: {"frame", "cap", "losses": [l_0, l_1, ...],
"keyframe_ids": [kf_0, kf_1, ...], "n_new_pts": N, "unseen_ratio": U}. l_k is
the mapping loss BEFORE iteration k's step (same convention as IterTrace:
"the loss at k is evaluated at pose k"); kf_k is which keyframe/view that
iteration optimised. n_new_pts and unseen_ratio are the same two novelty
signals AdaptiveMapper.compute_budget() already uses - None if the caller
doesn't track one (e.g. frame 0, which has no densification signal by
convention, or a new-submap frame on Gaussian-SLAM, which skips novelty
entirely). unseen_ratio needs no scene-tuned reference constant (it's already
a fraction of pixels in [0,1]) - n_new_pts does (AdaptiveMapper.n_new_pts_ref),
and that reference has no self-calibration path, unlike the class's other two
signals - see fig_mapping_adaptive_headroom.py for why that matters.

WHY keyframe_ids MATTERS. Unlike tracking, mapping samples a random keyframe
EVERY iteration (see optimize_submap() / SplaTAM's mapping loop) - iteration
k and k+1 are very often different objectives entirely. The raw per-iteration
loss curve is therefore noisy by construction, not because the optimiser is
struggling: most of that noise is "which view", not "how converged". Recording
keyframe_ids lets a reader re-group by keyframe (loss vs. Nth-time-this-
keyframe-was-visited) to see the per-view convergence the raw curve hides.
"""

import json
import math
import os


class MapTrace:
    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        path = os.environ.get("MAP_ITER_TRACE", "").strip() or cfg.get("out_path", "")
        self.enabled = bool(path) and path not in ("0", "false", "False")
        self.path = path
        self.every = max(1, int(os.environ.get("MAP_ITER_TRACE_EVERY", "")
                                or cfg.get("every", 1)))
        self._frame = None
        self._cap = None
        self._losses = None
        self._keyframe_ids = None
        self._n_new_pts = None
        self._unseen_ratio = None
        self._fh = None
        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._fh = open(path, "w")
            print(f"[MapTrace] writing per-mapping-iteration loss to {path}, "
                 f"every {self.every} frame(s) (one host sync per traced "
                 "iteration - not a timed run)", flush=True)

    def wants(self, frame):
        return self.enabled and int(frame) % self.every == 0

    @property
    def active(self):
        return self.enabled and self._frame is not None

    def begin_frame(self, frame, cap, n_new_pts=None, unseen_ratio=None):
        if not self.enabled:
            return
        if int(frame) % self.every != 0:
            self._frame = None
            return
        self._frame = int(frame)
        self._cap = int(cap)
        self._losses = []
        self._keyframe_ids = []
        self._n_new_pts = (None if n_new_pts is None or not math.isfinite(n_new_pts)
                          else int(n_new_pts))
        self._unseen_ratio = (None if unseen_ratio is None or not math.isfinite(unseen_ratio)
                              else float(unseen_ratio))

    def capture(self, loss, keyframe_id=None):
        """loss: a scalar tensor or float, the mapping loss for the iteration
        about to be backpropagated (evaluated BEFORE the step, same
        convention as IterTrace). keyframe_id: which keyframe/view this
        iteration sampled (int-like or None if the caller doesn't track one)."""
        if not self.active:
            return
        self._losses.append(float(loss))
        self._keyframe_ids.append(None if keyframe_id is None else int(keyframe_id))

    def end_frame(self):
        if not self.active:
            return
        self._fh.write(json.dumps(
            {"frame": self._frame, "cap": self._cap, "losses": self._losses,
             "keyframe_ids": self._keyframe_ids, "n_new_pts": self._n_new_pts,
             "unseen_ratio": self._unseen_ratio}) + "\n")
        self._fh.flush()
        self._frame = None
