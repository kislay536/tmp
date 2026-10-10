"""
grad_variance.py - is the tile-masked pose gradient a good estimator?

WHAT QUESTION THIS ANSWERS, AND WHY IT IS NOT A NEW METHOD.

The ladder records an unexplained result: tile masking at sample_ratio 0.75
IMPROVED ATE (turning it off was worse on both axes, p=0.071, the closest to
separation anything on the branch reached), while 0.6 was worse than 0.75.
The hypothesis written down at the time was:

    "rendering 75% of tiles makes each pose gradient a subsample, which is
     stochastic-gradient-like and may regularise. That is a hypothesis with
     no evidence behind it."

This measures it. It is a DIAGNOSTIC, not a contribution - sparse-pixel
tracking is long established in direct VO (DSO tracks on ~2000 points) and is
already used in 3DGS-SLAM. What is missing is not the technique but an
explanation of our own operating point, and "we cannot say why our best config
works" is a weak spot in a defence.

WHAT IT MEASURES. At chosen frames and iterations, the full-render pose
gradient g_full is computed once, then N masked gradients g_i are computed per
(sample_ratio, gradient_frac) setting. From those:

    bias       cos(mean(g_i), g_full)   direction agreement
               |mean(g_i)| / |g_full|   magnitude agreement
    variance   mean ||g_i - mean(g_i)|| / |mean(g_i)|      relative spread
    SNR        |mean(g_i)| / mean||g_i - mean(g_i)||

A subsampled estimator can be unbiased and still useless if its variance
swamps the signal, and it can be biased and still fine if the bias points the
same way. Both numbers are needed; neither alone decides anything.

THE CONFOUND THIS EXISTS TO SEPARATE. build_tile_mask is NOT uniform sampling.
It keeps the top `gradient_frac` of tiles by image-gradient magnitude and fills
the rest uniformly at random, so it is already an IMPORTANCE-SAMPLED estimator
with variance reduction built in. Sweeping sample_ratio alone - which is what
the 0.6-vs-0.75 comparison did - moves the sample size AND the
importance/uniform split together. This sweeps them separately.

    gradient_frac = 0.0   pure uniform -> the clean estimator-variance baseline
    gradient_frac = 0.5   the shipped configuration
    gradient_frac = 1.0   pure top-gradient, no randomness -> variance ~ 0,
                          bias is whatever edge tiles alone give you

If the shipped setting turns out to be low-variance because it is half
deterministic, then the "stochastic regularisation" story is WRONG and the
0.75 result needs a different explanation. That is a real possible outcome and
the probe is built so it can produce it.

COST. Each probe iteration runs 1 + N x len(ratios) x len(gradient_fracs)
extra forward+backward passes. At the defaults that is 1 + 5x3x3 = 46 extra
iterations on a probed iteration, so probe FEW iterations on FEW frames. The
defaults touch 3 frames x 3 iterations = 9 probed iterations per run.

IT MUST NOT PERTURB THE RUN.
  - gradients are saved and restored around every probe, so the real
    iteration's backward is unaffected
  - the optimizer is never stepped and never sees the probe
  - the probe runs on a COPY of nothing: it re-renders from the SAME live pose,
    which is read-only here
  - it is default OFF and refuses to run under an active CUDA graph capture,
    because a probe pass launching into a non-capturing stream is exactly the
    cudaStreamCaptureModeGlobal error the prefetch contract documents

MODEL-AGNOSTIC BY CONSTRUCTION. The call site supplies a closure that does one
forward+backward for a given tile mask and leaves gradients on the pose
tensors. Nothing here knows about SplaTAM, MonoGS or Gaussian-SLAM, so the
same probe answers the same question on all three - which is the point, since
the interesting result is whether the curves DIFFER between them.
"""

from __future__ import annotations

import json
import math
import os


def _flat_pose_grad(pose_tensors):
    """Concatenate the pose parameters' gradients into one vector.

    Returns None if any gradient is missing, which is a real condition (a
    backward that produced nothing) and must not be silently read as zeros.
    """
    import torch

    parts = []
    for t in pose_tensors:
        if t is None or t.grad is None:
            return None
        parts.append(t.grad.detach().reshape(-1).float())
    if not parts:
        return None
    return torch.cat(parts)


class GradVarianceProbe:
    """Measures bias and variance of the tile-masked pose gradient."""

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", False))
        self.frames = list(cfg.get("frames", [50, 200, 400]))
        self.iters = list(cfg.get("iters", [10, 50, 100]))
        self.repeats = int(cfg.get("repeats", 5))
        self.sample_ratios = list(cfg.get("sample_ratios", [0.5, 0.75, 0.9]))
        self.gradient_fracs = list(cfg.get("gradient_fracs", [0.0, 0.5, 1.0]))
        self.out_path = cfg.get("out_path", "grad_variance.jsonl")
        self._records = []
        self._probed = 0

        print(
            f"[GradVar] constructed (enabled={self.enabled}, "
            f"frames={self.frames}, iters={self.iters}, repeats={self.repeats}, "
            f"sample_ratios={self.sample_ratios}, "
            f"gradient_fracs={self.gradient_fracs})",
            flush=True,
        )

    def should_probe(self, time_idx: int, iter_idx: int) -> bool:
        return self.enabled and time_idx in self.frames and iter_idx in self.iters

    # -- the measurement ---------------------------------------------------

    def run(self, time_idx, iter_idx, pose_tensors, backward_fn,
            mask_fn, base_cfg):
        """Measure one (frame, iteration) point.

        pose_tensors  list of tensors whose .grad forms the pose gradient
        backward_fn   callable(tile_mask) -> float. Must zero the pose grads,
                      run forward+backward once, and leave .grad populated.
                      Must NOT step an optimizer.
        mask_fn       callable(sample_ratio, gradient_frac) -> tile mask tensor
        base_cfg      the live pixel_sample config, for the record only
        """
        import torch

        if not self.enabled:
            return
        # is_current_stream_capturing() raises on a CPU-only torch build
        # ("Tried to instantiate dummy base class"), so gate it on CUDA being
        # present. On the VM this is a real check; off-GPU it is vacuous, which
        # is correct - there is no capture to collide with.
        capturing = False
        if torch.cuda.is_available():
            capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            # A probe pass here would launch into a non-capturing stream under
            # cudaStreamCaptureModeGlobal. Fail loudly rather than corrupt the
            # capture - the run is misconfigured, not merely unlucky.
            raise RuntimeError(
                "[GradVar] probe reached while a CUDA graph capture is active. "
                "Set tracking.iteration_graph.enabled=False for probe runs."
            )

        saved = [None if t.grad is None else t.grad.detach().clone()
                 for t in pose_tensors]

        try:
            backward_fn(None)                      # full render, no mask
            g_full = _flat_pose_grad(pose_tensors)
            if g_full is None:
                print(f"[GradVar] frame {time_idx} iter {iter_idx}: full render "
                      f"produced no pose gradient, skipping this point",
                      flush=True)
                return
            g_full = g_full.clone()
            full_norm = float(g_full.norm())

            for ratio in self.sample_ratios:
                for gfrac in self.gradient_fracs:
                    samples = []
                    for _ in range(self.repeats):
                        mask = mask_fn(ratio, gfrac)
                        backward_fn(mask)
                        g = _flat_pose_grad(pose_tensors)
                        if g is not None:
                            samples.append(g.clone())
                    if not samples:
                        continue
                    self._records.append(
                        self._summarise(time_idx, iter_idx, ratio, gfrac,
                                        samples, g_full, full_norm)
                    )
        finally:
            # Restore whatever the loop had before we touched anything.
            for t, s in zip(pose_tensors, saved):
                if s is None:
                    t.grad = None
                else:
                    if t.grad is None:
                        t.grad = s
                    else:
                        t.grad.copy_(s)

        self._probed += 1

    @staticmethod
    def _summarise(time_idx, iter_idx, ratio, gfrac, samples, g_full, full_norm):
        import torch

        stack = torch.stack(samples)             # (N, D)
        mean = stack.mean(dim=0)
        mean_norm = float(mean.norm())

        # bias: does the expected masked gradient point where the full one does
        if full_norm > 0 and mean_norm > 0:
            cos_full = float(torch.dot(mean, g_full) / (mean_norm * full_norm))
        else:
            cos_full = float("nan")

        # variance: how far individual draws sit from their own mean
        dev = (stack - mean).norm(dim=1)
        mean_dev = float(dev.mean())
        rel_spread = mean_dev / mean_norm if mean_norm > 0 else float("nan")
        snr = mean_norm / mean_dev if mean_dev > 0 else float("inf")

        # per-draw direction agreement with the FULL gradient, which is what
        # a single tracking iteration actually gets - the mean is never
        # observed by the optimiser
        if full_norm > 0:
            per_draw_cos = (stack @ g_full) / (stack.norm(dim=1) * full_norm)
            cos_draw_mean = float(per_draw_cos.mean())
            cos_draw_min = float(per_draw_cos.min())
        else:
            cos_draw_mean = cos_draw_min = float("nan")

        return {
            "frame": int(time_idx),
            "iter": int(iter_idx),
            "sample_ratio": float(ratio),
            "gradient_frac": float(gfrac),
            "n": int(stack.shape[0]),
            "full_norm": full_norm,
            "mean_norm": mean_norm,
            "mag_ratio": mean_norm / full_norm if full_norm > 0 else float("nan"),
            "cos_mean_vs_full": cos_full,
            "cos_draw_vs_full_mean": cos_draw_mean,
            "cos_draw_vs_full_min": cos_draw_min,
            "rel_spread": rel_spread,
            "snr": snr,
        }

    # -- reporting ---------------------------------------------------------

    def save(self):
        if not self.enabled or not self._records:
            return
        try:
            with open(self.out_path, "w", encoding="utf-8") as fh:
                for r in self._records:
                    fh.write(json.dumps(r) + "\n")
            print(f"[GradVar] wrote {len(self._records)} records to "
                  f"{self.out_path}", flush=True)
        except OSError as exc:
            print(f"[GradVar] could not write {self.out_path}: {exc}", flush=True)

    def summary(self) -> str:
        if not self.enabled:
            return "Gradient-variance probe: disabled"
        if not self._records:
            return "Gradient-variance probe: enabled but never fired"

        by_key = {}
        for r in self._records:
            by_key.setdefault((r["sample_ratio"], r["gradient_frac"]), []).append(r)

        lines = [
            f"Gradient-variance probe: {self._probed} points, "
            f"{len(self._records)} records",
            "  ratio  gfrac   cos(mean,full)  |mean|/|full|  rel spread   SNR"
            "   worst draw cos",
            "  " + "-" * 76,
        ]
        for (ratio, gfrac) in sorted(by_key):
            rs = by_key[(ratio, gfrac)]
            avg = lambda k: sum(x[k] for x in rs) / len(rs)  # noqa: E731
            worst = min(x["cos_draw_vs_full_min"] for x in rs)
            snr = avg("snr")
            # A fully deterministic mask (gradient_frac=1.0) leaves a spread of
            # float noise rather than exact zero, so isinf never fires and the
            # column blows out to eight digits. Anything past 1e6 is "no
            # variance" for our purposes.
            if math.isinf(snr) or snr > 1e6:
                snr_s = " >1e6"
            else:
                snr_s = f"{snr:5.2f}"
            lines.append(
                f"  {ratio:5.2f}  {gfrac:5.2f}   {avg('cos_mean_vs_full'):13.4f}  "
                f"{avg('mag_ratio'):13.4f}  {avg('rel_spread'):10.4f}  {snr_s}"
                f"   {worst:13.4f}"
            )
        lines.append("")
        lines.append("  HOW TO READ IT. cos(mean,full) near 1 with a large rel")
        lines.append("  spread is the stochastic-regularisation story: unbiased")
        lines.append("  direction, noisy draws. cos near 1 with a SMALL spread")
        lines.append("  means the mask is nearly deterministic and the 0.75")
        lines.append("  result is NOT about gradient noise - look elsewhere.")
        lines.append("  A mag_ratio well below 1 at gradient_frac=0 would mean")
        lines.append("  the estimator is scale-biased, which an optimiser sees")
        lines.append("  as an effective learning-rate change, not as noise.")
        return "\n".join(lines)
