"""Are the tiles a gradient mask KEEPS the expensive ones?

THE QUESTION. build_tile_mask keeps the top `gradient_frac` of tiles by image-
gradient magnitude and fills the rest uniformly. Masking 40% of TILES has
repeatedly failed to buy anything like 40% of the time: SplaTAM gains ~10%,
MonoGS 0.00, GSLAM -2.2% at 780k Gaussians, and masking the backward render as
well - which was genuinely not masked until now - added 0.08% on MonoGS and
0.2% on GSLAM.

The hypothesis this closes: HIGH IMAGE GRADIENT IS WHERE THE GAUSSIANS ARE.
Edges and texture are exactly the regions the mapper densifies; flat wall and
ceiling tiles are sparse. If so, the tiles the mask throws away are the CHEAP
ones - short point_list ranges, few rounds, little to load - and dropping 40%
of tiles drops far less than 40% of the work. That would explain every number
above with one mechanism, and it would mean the problem is the SELECTION RULE
rather than the masking.

HOW IT MEASURES THAT, WITHOUT nsys. Three timed forward+backward passes over
the same frame, same Gaussians, same everything but the mask:

    t_dense   no mask
    t_keep    the mask as shipped
    t_drop    its exact complement

Write D for dense, U for work no tile mask can shrink (preprocess, binning,
sort - all per-Gaussian or per-intersection), and R for the render stages it
can. Every tile is in exactly one of keep/drop, so

    D      = U + R
    t_keep = U + R_keep
    t_drop = U + R_drop          with  R_keep + R_drop = R

Three equations, three unknowns:

    U      = t_keep + t_drop - D
    R      = D - U
    R_keep = t_keep - U

So this reports BOTH numbers the nsys stage split was built for - the
unmaskable share U/D, and the share of render the kept tiles actually carry -
from three wall timings and no profiler. The identity needs masked tiles to
cost ~nothing, which is true only with the backward mask in place; with
DGR_MASK_BACKWARD=0 a masked tile still pays its full range in the backward and
U comes out inflated. The probe checks that and says so.

WHAT WOULD REFUTE THE HYPOTHESIS. If R_keep/R lands at the kept-tile fraction
(0.6 as shipped), tiles cost the same on average wherever they are, the mask is
an honest 40% cut of the render stage, and the nulls have to be explained by U
being large instead. That is a real possible outcome and the rule below is
written to report it as such.

NOT A TIMED RUN. Each probe point runs 3 x repeats extra forward+backward
passes. Nothing about the host run's wall time, in-loop time or ms/iter means
anything with this enabled, exactly as for GradVarianceProbe. The ONLY output
is the [TileCost] table.
"""

import json
import os

import torch

from utils.pixel_sample import build_tile_mask, kept_pixel_fraction, tile_dims


class TileCostProbe:
    """Times dense / kept / dropped renders and solves for where the cost is."""

    def __init__(self, cfg: dict):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.frames = [int(f) for f in cfg.get("frames", [50, 200, 400])]
        self.iters = [int(i) for i in cfg.get("iters", [10, 50])]
        # 15, NOT 5. The first run showed cost_share_kept at 0.545 and 0.900
        # on ADJACENT probe points of the same frame - a spread wider than
        # the entire range the decision rule discriminates over. The raw
        # ratios were stable; the difference-of-differences was not, because
        # it is extracted from ~0.2-2 ms gaps on 10-14 ms measurements.
        self.repeats = int(cfg.get("repeats", 15))
        self.warmup = int(cfg.get("warmup", 2))
        self.out_path = cfg.get("out_path", "tile_cost.jsonl")
        self.rows = []

        if self.repeats < 3:
            # The median of fewer than three is one draw wearing a hat.
            raise ValueError("tile cost probe repeats must be >= 3")

        if self.enabled:
            print(f"[TileCost] constructed (frames={self.frames}, "
                  f"iters={self.iters}, repeats={self.repeats})", flush=True)

    def should_probe(self, time_idx: int, iter_idx: int) -> bool:
        return (self.enabled and time_idx in self.frames
                and iter_idx in self.iters)

    @staticmethod
    def _mask_backward_on():
        """Whether the installed rasterizer masks the BACKWARD render too.

        The decomposition below assumes a masked tile costs ~nothing. That is
        false when only the forward is masked - the backward still walks the
        tile's whole Gaussian range - and U then absorbs that cost and reads
        far too high. Worth stating per run rather than assuming.
        """
        try:
            import diff_gaussian_rasterization as dgr
            return bool(getattr(dgr, "_MASK_BACKWARD", False))
        except Exception:
            return False

    def _time(self, fn, mask):
        """Median ms over `repeats`, after `warmup`, on CUDA events.

        Median not mean: one preemption or one clock excursion moves a mean and
        does not move a median, and this probe runs inside a live SLAM process
        with a mapper competing for the GPU.
        """
        for _ in range(self.warmup):
            fn(mask)
        torch.cuda.synchronize()
        samples = []
        for _ in range(self.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(mask)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        samples.sort()
        return samples[len(samples) // 2]

    def run(self, time_idx, iter_idx, backward_fn, gt_image, H, W, ps_cfg,
            pose_tensors=None):
        """One probe point. backward_fn(mask) runs an ISOLATED fwd+bwd.

        pose_tensors MUST be given whenever the host loop accumulates into
        .grad. backward_fn leaves gradients populated from the LAST probe
        pass, and SplaTAM's real iteration calls _loss.backward() without
        zeroing first - it zeroes with .zero_() inside the step instead - so
        a probe gradient left lying there is ADDED to the next real one.
        That corrupts the pose gradient on exactly the probed iterations and
        reports nothing: the run continues, converges slightly worse, and
        the only symptom is an ATE. GradVarianceProbe guards this the same
        way; the guard was missing here.
        """
        if not self.enabled:
            return

        saved = None
        if pose_tensors:
            saved = [None if t.grad is None else t.grad.detach().clone()
                     for t in pose_tensors]
        try:
            self._measure(time_idx, iter_idx, backward_fn, gt_image, H, W,
                          ps_cfg)
        finally:
            if saved is not None:
                for t, s in zip(pose_tensors, saved):
                    if s is None:
                        t.grad = None
                    elif t.grad is None:
                        t.grad = s
                    else:
                        t.grad.copy_(s)

    def _measure(self, time_idx, iter_idx, backward_fn, gt_image, H, W,
                 ps_cfg):
        mask = build_tile_mask(gt_image, H, W, ps_cfg)
        n_tiles = int(mask.numel())
        kept_tiles = float(mask.sum().item()) / max(n_tiles, 1)
        kept_pixels = float(kept_pixel_fraction(mask, H, W).item())

        t_dense = self._time(backward_fn, None)
        t_keep = self._time(backward_fn, mask)
        t_drop = self._time(backward_fn, ~mask)

        # THE CEILING FOR THE WHOLE SPARSE FAMILY, in one number. With every
        # tile masked, the render stages do nothing at all, so
        #
        #     (t_dense - t_allmasked) / t_dense
        #
        # is the ENTIRE cost any tile-level scheme could ever remove -
        # whatever it keeps, however it chooses, at the render stage.
        # No selection rule, no schedule and no compaction can beat it.
        #
        # It does NOT bound a mask applied EARLIER. Masking at
        # preprocessCUDA's tiles_touched would also drop key emission, the
        # radix sort and identifyTileRanges, none of which this removes: an
        # all-masked render still emits and sorts every key. So read this as
        # the render-stage ceiling, and price the binning stages separately
        # (profiling/sparse_stage_split.py buckets them).
        try:
            t_allmasked = self._time(backward_fn, torch.zeros_like(mask))
        except Exception as exc:
            # An all-masked render is a degenerate input - a loss rescaled
            # by a kept fraction of zero would be inf. Losing this one
            # number must not lose the probe point that carries the others.
            print(f"[TileCost] all-masked timing failed ({exc}); "
                  f"continuing without the family ceiling", flush=True)
            t_allmasked = float("nan")

        # U = t_keep + t_drop - D. Clamped at zero: the identity can go
        # slightly negative on noise when U is genuinely tiny, and a negative
        # "unmaskable share" would be reported as a finding by someone reading
        # the table quickly.
        u = max(t_keep + t_drop - t_dense, 0.0)
        r = max(t_dense - u, 1e-9)
        r_keep = max(t_keep - u, 0.0)
        cost_share_kept = min(r_keep / r, 1.0)

        row = {
            "frame": int(time_idx), "iter": int(iter_idx),
            "tile_grid": list(tile_dims()), "n_tiles": n_tiles,
            "kept_tiles": kept_tiles, "kept_pixels": kept_pixels,
            "t_dense_ms": t_dense, "t_keep_ms": t_keep, "t_drop_ms": t_drop,
            "t_allmasked_ms": t_allmasked,
            "render_stage_ceiling": ((t_dense - t_allmasked) / t_dense
                                     if t_allmasked == t_allmasked else None),
            "unmaskable_share": u / max(t_dense, 1e-9),
            "cost_share_kept": cost_share_kept,
            "mask_backward": self._mask_backward_on(),
            "sample_ratio": float(ps_cfg.get("sample_ratio", 0.6)),
            "gradient_frac": float(ps_cfg.get("gradient_frac", 0.5)),
        }
        self.rows.append(row)

        ceiling = row["render_stage_ceiling"]
        print(f"[TileCost] frame {time_idx} iter {iter_idx}: "
              f"tiles kept {kept_tiles:.3f}, "
              f"dense/keep/drop/allmasked "
              f"{t_dense:.3f}/{t_keep:.3f}/{t_drop:.3f}/{t_allmasked:.3f} ms, "
              f"RENDER-STAGE CEILING "
              f"{('%.3f' % ceiling) if ceiling is not None else 'n/a'}, "
              f"cost share kept {cost_share_kept:.3f}", flush=True)

        if self.out_path:
            try:
                with open(self.out_path, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
            except OSError:
                pass

    def summary(self) -> str:
        if not self.enabled:
            return "Tile cost probe: disabled"
        if not self.rows:
            return "Tile cost probe: enabled but never fired"

        def med(key):
            vals = sorted(r[key] for r in self.rows)
            return vals[len(vals) // 2]

        kept = med("kept_tiles")
        share = med("cost_share_kept")
        unmask = med("unmaskable_share")
        excess = share - kept

        ceilings = [r["render_stage_ceiling"] for r in self.rows
                    if r.get("render_stage_ceiling") is not None]
        ceiling = (sorted(ceilings)[len(ceilings) // 2] if ceilings else None)

        lines = [
            f"Tile cost probe: {len(self.rows)} points",
            f"  tiles kept            {kept:.3f}",
            f"  pixels kept           {med('kept_pixels'):.3f}",
            f"  COST share kept       {share:.3f}   (excess over tiles: "
            f"{excess:+.3f})",
            f"  unmaskable share U/D  {unmask:.3f}",
        ]

        if ceiling is not None:
            lines.append("")
            lines.append(
                f"  RENDER-STAGE CEILING  {ceiling:.3f}  <- the MOST any "
                f"tile-level scheme can remove")
            lines.append(
                f"    Measured with EVERY tile masked, so no selection rule, "
                f"schedule or")
            lines.append(
                f"    launch compaction can beat it. Best case speedup "
                f"{1.0 / max(1.0 - ceiling, 1e-9):.2f}x.")
            lines.append(
                f"    Does NOT bound a mask applied at preprocessCUDA: an "
                f"all-masked render")
            lines.append(
                f"    still emits and sorts every key. Price binning "
                f"separately.")

        if not all(r["mask_backward"] for r in self.rows):
            lines.append(
                "  !! DGR_MASK_BACKWARD was OFF for some or all points. A "
                "masked tile still paid its full backward range, so U is "
                "INFLATED and the split below understates the render stage. "
                "Re-run with it on before reading any of this.")

        # Pre-registered. Written before the run; do not edit to fit a result.
        lines.append("")
        lines.append("  DECISION RULE (pre-registered)")
        if excess >= 0.15:
            lines.append(
                f"    CONFIRMED. The kept tiles carry {share:.0%} of render "
                f"cost while being {kept:.0%} of tiles. The mask discards the "
                "CHEAP tiles, so its ceiling is not (1 - kept) x render but "
                f"({1 - share:.2f}) x render. The problem is the SELECTION "
                "RULE, not the masking.")
        elif excess <= 0.05:
            lines.append(
                f"    REFUTED. Cost share {share:.3f} tracks tile share "
                f"{kept:.3f}, so tiles cost about the same wherever they are "
                "and the mask is an honest cut of the render stage. The nulls "
                f"must then come from U, which is {unmask:.0%} of the time "
                "here - read that number, not this one.")
        else:
            lines.append(
                f"    MARGINAL ({excess:+.3f}). Between the 0.05 and 0.15 "
                "bars. Add probe points before concluding either way.")
        return "\n".join(lines)
