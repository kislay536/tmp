"""Measure how much of the Gaussian set is actually visible per tracking frame,
and how much that set CHANGES within a frame. Shared by SplaTAM, MonoGS and
Gaussian-SLAM.

WHY THIS EXISTS. Two proposed optimisations depend on numbers nobody has
measured:

  cov3D precompute        hoists computeCov3D out of the tracking iteration.
                          Its ceiling is preprocessCUDA's share of kernel time
                          (that is profiling/kernel_share.py's job, not this
                          file's).

  active-set compaction   runs the frame's ~120 tracking iterations on only the
                          Gaussians inside the frustum. Its ceiling is the
                          VISIBLE FRACTION, and its SAFETY is decided by the
                          within-frame CHURN. Both are measured here.

If the visible fraction comes back near 1.0 there is nothing to compact and the
idea is dead for the cost of one run. That is the point.

  ============================================================
  WHY CHURN IS THE NUMBER THAT MATTERS, NOT THE FRACTION
  ============================================================

  Compaction freezes a subset at frame start and renders from it for the whole
  frame. If a Gaussian becomes visible mid-frame and is not in that subset, it
  silently does not render. No error, no fallback, one wrong frame, and the
  damage surfaces later as trajectory drift that looks like something else.

  That is exactly the shape of the binning-capacity overflow this repo already
  documents: the first overflow was the real failure, everything after it
  (capacity ratchet, observed_max collapse, OOM) was consequence. The fix there
  was a measured margin plus a hard error on first overflow, and compaction
  needs the same discipline - which means measuring the churn BEFORE writing
  the optimisation, not after it produces a mysterious ATE.

  new_frac (below) is that number: of the Gaussians visible at the END of a
  tracking frame, what fraction were NOT visible at the start. It sizes the
  frustum dilation the same way observed_max sizes the binning margin.

  ============================================================
  IT USES radii, NOT markVisible
  ============================================================

  `radii > 0` is the rasterizer's own record of which Gaussians survived
  preprocessing and got a screen-space footprint. Every model's render already
  returns it, so this costs no extra kernel and cannot disagree with the
  definition the optimisation would actually use. markVisible would be a
  second, reimplemented notion of visibility with its own frustum test, which
  is how you end up measuring something adjacent to the thing you care about.

  ============================================================
  NO SYNCS IN THE LOOP
  ============================================================

  Every statistic accumulates into device-side scalars and is read back ONCE,
  in summary(), at the end of the run. A .item() per frame would be ~0.5ms on
  the measurements in this repo - negligible against per-frame compute, but
  this is instrumentation for a branch whose whole argument is about host
  syncs, and an instrument that perturbs the thing it measures is how you get a
  confident wrong answer.

  Calls are made at frame boundaries, OUTSIDE any captured region. Do not call
  record_* from inside a CUDA graph capture: the accumulators would be baked
  into the replay and every frame would re-add the capture-time values.

USAGE (identical on all three models):

    probe = VisibilityProbe(enabled=cfg.get('visibility_probe', False))
    ...
    for it in range(num_iters):
        im, radii, *_ = render(...)
        if it == 0:
            probe.record_first(radii)
        last_radii = radii
    probe.record_last(last_radii)     # after the tracking loop, per frame
    ...
    print(probe.summary())            # once, at end of run
"""

import torch


class VisibilityProbe:
    """Per-frame visible-fraction and within-frame churn statistics.

    Disabled by default. When disabled every method is a no-op returning
    immediately, so call sites can be left in place unconditionally.
    """

    def __init__(self, enabled=False, device="cuda"):
        self.enabled = enabled
        if not enabled:
            return

        self.device = device
        # Device-side accumulators. Drained once, in summary().
        z = lambda: torch.zeros((), dtype=torch.float64, device=device)
        self._n_frames = z()
        self._sum_P = z()
        self._sum_visible = z()
        self._sum_visible_frac = z()
        self._sum_new_frac = z()
        self._max_new_frac = z()
        self._sum_lost_frac = z()
        self._n_size_mismatch = z()
        self._n_unpaired = z()

        self._first_mask = None

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------

    def record_first(self, radii):
        """Record the visible set at the FIRST tracking iteration of a frame."""
        if not self.enabled:
            return
        if self._first_mask is not None:
            # record_last was never called for the previous frame - the caller
            # returned early (early stopping at iteration 0, a skipped frame).
            # Count it rather than silently pairing across a frame boundary.
            self._n_unpaired += 1
        self._first_mask = (radii > 0).detach().clone()

    def record_last(self, radii):
        """Record the visible set at the LAST tracking iteration and fold the
        frame's statistics into the accumulators."""
        if not self.enabled:
            return
        first = self._first_mask
        self._first_mask = None
        if first is None:
            self._n_unpaired += 1
            return

        last = radii > 0
        if last.shape != first.shape:
            # The Gaussian count changed mid-frame. That should not happen
            # during tracking (densification runs in mapping), and if it does,
            # compaction's premise is broken for this frame - so it is counted
            # and excluded rather than reconciled.
            self._n_size_mismatch += 1
            return

        P = first.numel()
        n_first = first.sum(dtype=torch.float64)
        n_last = last.sum(dtype=torch.float64)
        # Newly visible: in `last`, absent from `first`. This is the quantity a
        # dilated frustum has to cover.
        n_new = (last & ~first).sum(dtype=torch.float64)
        n_lost = (first & ~last).sum(dtype=torch.float64)

        denom = torch.clamp(n_first, min=1.0)
        new_frac = n_new / denom
        lost_frac = n_lost / denom

        self._n_frames += 1
        self._sum_P += float(P)
        self._sum_visible += n_first
        self._sum_visible_frac += n_first / max(P, 1)
        self._sum_new_frac += new_frac
        self._sum_lost_frac += lost_frac
        self._max_new_frac = torch.maximum(self._max_new_frac, new_frac)
        # n_last is folded in only through new/lost; kept out of the averages
        # so `visible_frac` means "at frame start", which is what compaction
        # would size its buffer from.
        del n_last

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def stats(self):
        """Drain the accumulators to the host. ONE sync, at end of run."""
        if not self.enabled:
            return None
        vals = torch.stack([
            self._n_frames, self._sum_P, self._sum_visible,
            self._sum_visible_frac, self._sum_new_frac, self._max_new_frac,
            self._sum_lost_frac, self._n_size_mismatch, self._n_unpaired,
        ]).cpu().tolist()
        (n, sum_P, sum_vis, sum_vis_frac,
         sum_new, max_new, sum_lost, n_mismatch, n_unpaired) = vals
        if n == 0:
            return {"frames": 0, "size_mismatch": int(n_mismatch),
                    "unpaired": int(n_unpaired)}
        return {
            "frames": int(n),
            "mean_gaussians": sum_P / n,
            "mean_visible": sum_vis / n,
            "visible_frac": sum_vis_frac / n,
            "new_frac_mean": sum_new / n,
            "new_frac_max": max_new,
            "lost_frac_mean": sum_lost / n,
            "size_mismatch": int(n_mismatch),
            "unpaired": int(n_unpaired),
        }

    def summary(self):
        if not self.enabled:
            return "[VisibilityProbe] disabled"
        s = self.stats()
        if s["frames"] == 0:
            return ("[VisibilityProbe] no complete frames recorded "
                    f"(size_mismatch={s['size_mismatch']}, "
                    f"unpaired={s['unpaired']}) - check the call sites")

        # Suggested dilation, in the same spirit as the binning margin: cover
        # the WORST frame observed, not the mean, then leave headroom. The
        # margin is on the compact buffer's CAPACITY, not on the frustum
        # geometry - a geometric dilation has to be chosen from the scene, but
        # the capacity is what silently overflows.
        margin = 1.0 + max(s["new_frac_max"], 0.0)
        lines = [
            "[VisibilityProbe] frames={frames}  mean Gaussians={mean_gaussians:,.0f}".format(**s),
            "  visible at frame start : {:.1%}  ({:,.0f} of {:,.0f})".format(
                s["visible_frac"], s["mean_visible"], s["mean_gaussians"]),
            "  newly visible by frame end : mean {:.3%}  MAX {:.3%}".format(
                s["new_frac_mean"], s["new_frac_max"]),
            "  no longer visible by end   : mean {:.3%}".format(s["lost_frac_mean"]),
            "  -> compaction ceiling  : {:.1%} of preprocess/binning work removable".format(
                1.0 - s["visible_frac"]),
            "  -> capacity margin >= {:.3f} to cover the worst frame seen".format(margin),
        ]
        if s["size_mismatch"]:
            lines.append("  WARNING: {} frames changed Gaussian count MID-FRAME "
                         "and were excluded. Compaction's premise does not hold "
                         "on those.".format(s["size_mismatch"]))
        if s["unpaired"]:
            lines.append("  note: {} unpaired record_first/record_last calls "
                         "(frames that exited the tracking loop early)".format(
                             s["unpaired"]))
        return "\n".join(lines)
