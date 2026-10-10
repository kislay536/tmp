"""How fast does the pose gradient go stale?

THE QUESTION. Tracking runs ~27 iterations per frame on a FROZEN map, and each
one pays a full render and a full backward to recover six numbers. If the
gradient at iteration k is still usable at k+1, half those renders are
unnecessary - and with them the preprocess, the binning and the sort, since a
skipped iteration skips the whole pipeline rather than one stage of it.

WHY THIS IS NOT THE SAME AS "REUSE THE SORT". Binning reuse was measured null
on this branch because the sort is ~6.7% of kernel time. Skipping the render
outright removes 100% of that iteration's pipeline, so it subsumes the reuse
idea and is worth far more if the gradient survives.

WHY A NAIVE ATTEMPT FAILS, AND WHAT THE TWEAK IS. The preconditioner reports
WANDER path/net around 3.2 on this model - the optimiser travels three times
its net displacement, so it is oscillating and partially cancelling. Reusing a
stale gradient in that regime doubles down on a direction that was about to
reverse. The tweak is to EXTRAPOLATE rather than reuse:

    g(xi + dxi)  ~  g(xi) + alpha * M @ dxi

where M is the preconditioner's existing 6x6, refactored every 10 steps and
already sitting in memory. A 6x6 matvec against a render is free.

THE SCALE PROBLEM, AND WHY THIS REPORTS AN UPPER BOUND. M is a gradient SECOND
MOMENT (beta2-decayed outer products), not a Hessian. For a least-squares
problem it is proportional to the Gauss-Newton Hessian, but the constant is
unknown, and the extrapolation above needs the right scale. So this scans alpha
and reports the BEST cosine any scale could achieve. That is deliberately
optimistic: no runtime scheme gets an oracle alpha. It is the correct quantity
for a go/no-go, because if the best possible extrapolation does not rescue the
gradient, no practical one will.

WHAT IT MEASURES. At a probe iteration it captures g0, lets the real loop run,
and re-measures the true gradient 1, 2 and 4 steps later:

    cos(g0, g_true)          does the DIRECTION survive
    |g0| / |g_true|          does the MAGNITUDE survive
    best cos with M          does curvature correction rescue it

NOT A TIMED RUN. Each probe point costs one extra forward+backward at the start
and one at each offset. Wall time, in-loop time and ms/iter are meaningless
with this on, exactly as for GradVarianceProbe.
"""

import json

import torch


def _flat_grad(tensors):
    parts = [t.grad.reshape(-1) for t in tensors if t.grad is not None]
    return torch.cat(parts) if parts else None


class GradStalenessProbe:
    """Measures gradient survival across real optimiser steps."""

    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.frames = [int(f) for f in cfg.get("frames", [50, 150, 250])]
        self.starts = [int(i) for i in cfg.get("starts", [5, 15])]
        self.offsets = sorted(int(o) for o in cfg.get("offsets", [1, 2, 4]))
        self.out_path = cfg.get("out_path", "grad_staleness.jsonl")
        # cos at or above this counts the gradient as still usable.
        self.usable_cos = float(cfg.get("usable_cos", 0.95))

        if any(o < 1 for o in self.offsets):
            raise ValueError("grad staleness offsets must be >= 1")

        self._active = None      # {start, g0, xi0}
        self._space = None       # which space g0 was measured in
        self._vectors = []       # (offset, g0, M@dxi, g_true) for shared-alpha
        self.rows = []

        if self.enabled:
            print(f"[GradStale] constructed (frames={self.frames}, "
                  f"starts={self.starts}, offsets={self.offsets})", flush=True)

    # -- scheduling --------------------------------------------------------

    def wants(self, time_idx, iter_idx):
        """True on any iteration this probe needs an isolated backward for.

        The host loop builds that backward closure conditionally, so it has to
        know in advance. Exact rather than conservative: a probe that quietly
        wanted more iterations than it declared would silently add passes to
        the run it is measuring.
        """
        if not self.enabled or time_idx not in self.frames:
            return False
        if iter_idx in self.starts:
            return True
        if self._active is not None and self._active["frame"] == time_idx:
            return (iter_idx - self._active["start"]) in self.offsets
        return False

    def reset_frame(self):
        self._active = None

    # -- measurement -------------------------------------------------------

    def observe(self, time_idx, iter_idx, pose_tensors, backward_fn,
                tangent_fn=None, curvature_fn=None, grad_map_fn=None):
        """Call on any iteration where wants() returned True.

        backward_fn(None) must run an ISOLATED forward+backward leaving .grad
        populated. Gradients are saved and restored around every call: the host
        loop accumulates into .grad without zeroing first, so a probe gradient
        left behind is added to the next real one - a silent pose-gradient
        corruption whose only symptom is a worse ATE.
        """
        if not self.enabled:
            return

        saved = [None if t.grad is None else t.grad.detach().clone()
                 for t in pose_tensors]
        try:
            backward_fn(None)
            # MEASURE IN THE SPACE THE OPTIMISER ACTUALLY STEPS IN.
            #
            # Without grad_map_fn this reads the raw parameter grads,
            # which for SplaTAM is a 7-vector: quaternion (4) plus
            # translation (3). That is the wrong space twice over.
            #
            # The quaternion is normalised before use, so the gradient
            # component ALONG q is a gauge direction that moves no pose at
            # all. It still enters a 7-D cosine, and if it is large and
            # noisy it drags the number down while meaning nothing - so a
            # 7-D cosine UNDERSTATES how well the pose gradient survives.
            #
            # And the curvature term below needs a 6-vector: M is 6x6, in
            # the SE(3) tangent. A 7-vector silently fails the shape check
            # and the extrapolated column reads n/a for the whole run,
            # which is exactly what happened the first time this ran.
            g = (grad_map_fn() if grad_map_fn is not None
                 else _flat_grad(pose_tensors))
            if g is None:
                return
            g = torch.as_tensor(g).reshape(-1)
            g = g.detach().clone().double()
            xi = None
            if tangent_fn is not None:
                xi = torch.as_tensor(tangent_fn(), dtype=torch.float64,
                                     device=g.device).reshape(-1)

            # CLOSE THE ACTIVE WINDOW FIRST, WITH THIS SAME FRESH g, BEFORE
            # this iteration is (possibly) also treated as a new start below.
            #
            # THIS ORDERING IS WHAT LETS ADJACENT STARTS SHARE A GRADIENT.
            # Iteration k+1 both closes window (k, k+1) AND can become the
            # very next window's g0 - so a caller wanting EVERY consecutive
            # pair (g1,g2), (g2,g3), (g3,g4), ... , not just every other one,
            # can set starts=range(1, N) directly instead of odd-only
            # spacing. The OLD code checked `iter_idx in self.starts` first
            # and returned immediately, so a start that coincided with a
            # pending close silently dropped that measurement - back-to-back
            # integer starts with offsets=[1] lost most rows to this.
            act = self._active
            if act is not None and act["frame"] == time_idx:
                offset = iter_idx - act["start"]
                if offset in self.offsets:
                    self._record(time_idx, act, offset, g, xi, curvature_fn)
                    if offset == self.offsets[-1]:
                        self._active = None

            if iter_idx in self.starts:
                # A ZERO g0 MAKES EVERY NUMBER BELOW MEANINGLESS, and
                # flatteringly so: cos(g0, g) is undefined, while the
                # extrapolation g0 + alpha*M@dxi collapses to alpha*M@dxi and
                # can be scaled onto ANY target, reporting a perfect 1.0000.
                # An offline test hit exactly that and it read as success.
                if float(g.norm()) <= 0.0:
                    print(f"[GradStale] frame {time_idx} iter {iter_idx}: "
                          f"zero gradient at the window start, skipping "
                          f"(nothing to go stale)", flush=True)
                    return
                # A new probe window. self._active is None here unless the
                # PREVIOUS window is still genuinely incomplete (this
                # iteration did not land on one of ITS remaining offsets
                # above) - a real config error, not the closed-then-reopened
                # case the block above already handled.
                if self._active is not None:
                    print(f"[GradStale] frame {time_idx}: start at {iter_idx} "
                          f"overlaps an open window from "
                          f"{self._active['start']}; widen `starts`", flush=True)
                self._space = ("tangent" if grad_map_fn is not None
                               else "raw parameters")
                self._active = {"frame": time_idx, "start": iter_idx,
                                "g0": g, "xi0": xi}
        finally:
            for t, s in zip(pose_tensors, saved):
                if s is None:
                    t.grad = None
                elif t.grad is None:
                    t.grad = s
                else:
                    t.grad.copy_(s)

    @staticmethod
    def _cos(a, b):
        na, nb = float(a.norm()), float(b.norm())
        if na <= 0 or nb <= 0:
            return float("nan")
        return float(torch.dot(a, b) / (na * nb))

    def _best_extrapolated_cos(self, g0, u, g_true):
        """max over alpha of cos(g0 + alpha*u, g_true).

        Scanned, not solved. The closed form exists but a scan is six lines,
        cannot be wrong about a sign convention, and this runs a handful of
        times per sequence. Reported WITH the alpha that won it, because an
        alpha far outside a plausible range is a warning that the fit is
        meaningless rather than a result.
        """
        best, best_a = -1.0, 0.0
        for a in torch.logspace(-6, 3, 40, dtype=torch.float64).tolist():
            for s in (1.0, -1.0):
                c = self._cos(g0 + (s * a) * u, g_true)
                if c == c and c > best:
                    best, best_a = c, s * a
        return best, best_a

    def _record(self, time_idx, act, offset, g_true, xi, curvature_fn):
        g0 = act["g0"]
        if float(g_true.norm()) <= 0.0:
            print(f"[GradStale] frame {time_idx} +{offset}: zero TRUE gradient, "
                  f"skipping this offset", flush=True)
            return
        row = {
            "frame": int(time_idx), "start": int(act["start"]),
            "offset": int(offset),
            "cos_stale": self._cos(g0, g_true),
            "mag_ratio": float(g0.norm() / max(float(g_true.norm()), 1e-30)),
            "cos_extrap": None, "alpha": None, "mag_extrap": None,
            "cos_secant": None, "mag_secant": None, "alpha_secant": None,
        }

        # The tweak: extrapolate with the preconditioner's own 6x6 instead of
        # reusing g0 unchanged. Needs both the tangent travelled and M.
        if curvature_fn is not None and xi is not None and act["xi0"] is not None:
            M = curvature_fn()
            if M is not None:
                M = torch.as_tensor(M, dtype=torch.float64, device=g0.device)
                dxi = xi - act["xi0"]
                if M.shape == (g0.numel(), g0.numel()):
                    u = M @ dxi
                    # THE SECANT ALPHA - the estimator a runtime scheme can
                    # actually hold, and the one this probe should have
                    # tested first.
                    #
                    # A FIXED alpha was never going to work and the reason is
                    # mechanical, not mysterious: M is a decayed E[g g^T], so
                    # its scale tracks |g|^2, and the step profile shows |g|
                    # falling ~14x within a single frame. |g|^2 therefore
                    # moves ~200x inside one frame and more across frames as
                    # the map grows, while the true Hessian does not shrink
                    # with it. alpha ~ 1/c is SUPPOSED to move.
                    #
                    # So fit it from what just happened, Barzilai-Borwein
                    # style: alpha = <dg, u> / <u, u> minimises
                    # ||dg - alpha*u||. One inner product, no tuning, and it
                    # re-derives the local scale every time.
                    #
                    # Fitted at the FIRST offset and applied to the later
                    # ones, never scored on the data it was fitted to - that
                    # would be the oracle again, one layer down.
                    if offset == self.offsets[0]:
                        uu = float(torch.dot(u, u))
                        if uu > 0.0:
                            act["alpha_secant"] = float(
                                torch.dot(g_true - g0, u)) / uu
                    elif act.get("alpha_secant") is not None:
                        _as = act["alpha_secant"]
                        _pred = g0 + _as * u
                        row["cos_secant"] = self._cos(_pred, g_true)
                        row["mag_secant"] = float(
                            _pred.norm() / max(float(g_true.norm()), 1e-30))
                        row["alpha_secant"] = _as
                    c, a = self._best_extrapolated_cos(g0, u, g_true)
                    row["cos_extrap"], row["alpha"] = c, a
                    row["mag_extrap"] = float(
                        (g0 + a * u).norm()
                        / max(float(g_true.norm()), 1e-30))
                    # Kept so summary() can fit ONE alpha across every
                    # measurement. A per-measurement alpha is an oracle no
                    # runtime scheme has; a single shared one is roughly
                    # what a real implementation could hold, and the gap
                    # between the two is the honest cost of the idea.
                    self._vectors.append(
                        (int(offset), g0.clone(), u.clone(), g_true.clone()))

        self.rows.append(row)
        extra = ("" if row["cos_extrap"] is None
                 else f", extrap {row['cos_extrap']:.4f} (alpha {row['alpha']:.3g})")
        print(f"[GradStale] frame {time_idx} start {act['start']} +{offset}: "
              f"cos {row['cos_stale']:.4f}, mag {row['mag_ratio']:.3f}{extra}",
              flush=True)

        if self.out_path:
            try:
                with open(self.out_path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
            except OSError:
                pass

    def _fit_shared_alpha(self):
        """One alpha for every measurement, maximising the MEAN cosine.

        Scanned over the same grid as the per-measurement fit, so the two
        numbers are directly comparable and the gap between them is exactly
        what per-measurement tuning was worth.
        """
        best, best_a = -1.0, 0.0
        for a in torch.logspace(-6, 3, 40, dtype=torch.float64).tolist():
            for s in (1.0, -1.0):
                cs = [self._cos(g0 + (s * a) * u, gt)
                      for _o, g0, u, gt in self._vectors]
                cs = [c for c in cs if c == c]
                if not cs:
                    continue
                m = sum(cs) / len(cs)
                if m > best:
                    best, best_a = m, s * a
        return best_a, best

    # -- reporting ---------------------------------------------------------

    def summary(self):
        if not self.enabled:
            return "Gradient staleness probe: disabled"
        if not self.rows:
            return "Gradient staleness probe: enabled but never fired"

        def med(rows, key):
            vals = sorted(r[key] for r in rows if r[key] is not None)
            return vals[len(vals) // 2] if vals else None

        lines = [f"Gradient staleness probe: {len(self.rows)} measurements"
                 f"  [space: {self._space or 0}]",
                 "  offset   n   cos(stale)   |g0|/|g|   cos with M   mag w/ M"
                 "   cos SECANT  mag SECANT",
                 "  " + "-" * 88]
        by_offset = {}
        for r in self.rows:
            by_offset.setdefault(r["offset"], []).append(r)

        horizon = 0
        for off in sorted(by_offset):
            rows = by_offset[off]
            c = med(rows, "cos_stale")
            m = med(rows, "mag_ratio")
            e = med(rows, "cos_extrap")
            me = med(rows, "mag_extrap")
            cs = med(rows, "cos_secant")
            ms = med(rows, "mag_secant")
            lines.append("  %6d %3d %11.4f %10.3f %13s %10s %12s %11s"
                         % (off, len(rows), c, m,
                            "n/a" if e is None else "%.4f" % e,
                            "n/a" if me is None else "%.3f" % me,
                            "fitted" if cs is None else "%.4f" % cs,
                            "here" if ms is None else "%.3f" % ms))
            if c >= self.usable_cos:
                horizon = off

        if self._vectors:
            alphas = sorted(r["alpha"] for r in self.rows
                            if r["alpha"] is not None)
            shared_a, shared_cos = self._fit_shared_alpha()
            lines.append("")
            lines.append(
                "  ONE SHARED alpha across all %d measurements: %.4g"
                % (len(self._vectors), shared_a))
            lines.append(
                "    mean cos %.4f   (per-measurement oracle alphas span "
                "%.3g to %.3g)" % (shared_cos, alphas[0], alphas[-1]))
            lines.append(
                "    THIS is the implementable number. A runtime scheme "
                "estimates alpha once, not per step.")
            # BY MAGNITUDE, and with a sign check. alpha is routinely
            # NEGATIVE here - g(xi+d) = g(xi) - H d makes -1 the right
            # answer for an exact Hessian - so a positives-only span test
            # silently never fires. It did not fire on a synthetic case
            # built specifically to make it fire.
            mags = sorted(abs(a) for a in alphas if a != 0.0)
            if mags and mags[-1] / mags[0] > 10:
                lines.append(
                    "    !! oracle |alpha| spans %.0fx, so no fixed alpha "
                    "fits them all - read the shared number, not the "
                    "per-measurement one." % (mags[-1] / mags[0]))
            if len({a > 0 for a in alphas}) > 1:
                lines.append(
                    "    !! oracle alpha CHANGES SIGN across measurements. "
                    "No single correction can follow that; treat the "
                    "per-measurement column as meaningless here.")

        # The secant column is the headline, because it is the only one a
        # runtime scheme could produce. Scored at the SECOND offset, since
        # the first is where alpha was fitted.
        sec_rows = by_offset.get(self.offsets[1]) if len(self.offsets) > 1 else None
        sec_cos = med(sec_rows, "cos_secant") if sec_rows else None
        sec_raw = med(sec_rows, "cos_stale") if sec_rows else None
        if sec_cos is not None:
            lines.append("")
            lines.append(
                "  SECANT alpha, fitted on the previous observed change: "
                "cos %.4f at offset %d" % (sec_cos, self.offsets[1]))
            lines.append(
                "    against %.4f raw at the same offset. THIS is the "
                "implementable comparison -" % sec_raw)
            lines.append(
                "    no oracle, no global constant, one inner product per "
                "step.")

        one = by_offset.get(1)
        c1 = med(one, "cos_stale") if one else None
        # THE SHARED-alpha COSINE, NOT THE PER-MEASUREMENT ONE.
        #
        # The first real run reported per-measurement extrapolated cosines of
        # 0.9977 / 0.9764 / 0.9524 and this rule duly announced that the tweak
        # was the point - while the shared-alpha line printed directly above
        # it read 0.6701, WORSE than reusing the gradient raw.
        #
        # The per-measurement column was an oracle artefact. Oracle alphas
        # spanned 2.9e6, and with that much freedom a scalar aligns a
        # 6-vector with nearly anything. The magnitude column said so
        # independently: the alpha maximising cosine at offset 4 inflated the
        # gradient 1.27 MILLION times.
        #
        # So the verdict reads the implementable number. The oracle column
        # stays in the table as a diagnostic - a large gap between the two IS
        # the finding, namely that no fixed alpha exists.
        # PREFER THE SECANT. A fixed alpha is not the implementable number
        # any more - the secant is, and on real data they disagree hard:
        # 0.9434 secant against 0.6290 shared, at the same offset. Leaving
        # the rule on the shared figure had it call MARGINAL while the line
        # above it reported a working correction.
        _shared = self._fit_shared_alpha()[1] if self._vectors else None
        _sec = med(by_offset.get(self.offsets[1]), "cos_secant") \
            if len(self.offsets) > 1 and by_offset.get(self.offsets[1]) else None
        e1 = _sec if _sec is not None else _shared
        oracle1 = med(one, "cos_extrap") if one else None

        lines.append("")
        if e1 is None:
            lines.append("  !! THE EXTRAPOLATED COLUMN NEVER RAN, so the"
                         " curvature tweak is UNTESTED and any verdict")
            lines.append("     below covers naive reuse only. Usual cause:"
                         " the gradient is not in the 6-D tangent, so it")
            lines.append("     fails the shape check against M - pass"
                         " grad_map_fn.")
            lines.append("")
        lines.append("  DECISION RULE (pre-registered)")
        if c1 is None:
            lines.append("    No offset-1 measurement; nothing to decide on.")
        elif c1 >= self.usable_cos:
            lines.append(
                f"    VIABLE RAW. cos {c1:.4f} at one step stale, so skipping "
                f"every 2nd render needs no correction at all. Usable horizon "
                f"{horizon} step(s): renders drop by a factor of "
                f"{1 + horizon}.")
        elif e1 is not None and e1 >= self.usable_cos:
            lines.append(
                f"    THE TWEAK IS THE POINT. Raw cos {c1:.4f} is too stale to "
                f"reuse, but extrapolating with the preconditioner's M reaches "
                f"{e1:.4f} under a SINGLE SHARED alpha - so one estimate covers "
                f"every measurement and a runtime scheme can hold it. Check the "
                f"magnitude column before building: a right direction at the "
                f"wrong scale is a changed learning rate.")
        elif c1 >= 0.80:
            lines.append(
                f"    MARGINAL. cos {c1:.4f} raw, "
                f"{'no M correction measured' if e1 is None else '%.4f with one shared alpha (oracle %.4f)' % (e1, oracle1 or float('nan'))}"
                f". A partially stale gradient still descends, so this is not "
                f"dead - but it will cost iterations, and this branch has just "
                f"measured that trade going NEGATIVE for sparse: the stopping "
                f"rule spent more iterations than the cheaper ones saved. Price "
                f"it in tracking s/FRAME, never ms/iter.")
        else:
            lines.append(
                f"    DEAD. cos {c1:.4f} after a SINGLE step"
                + ("" if e1 is None else f", and one shared alpha reaches only "
                   f"{e1:.4f}")
                + ". The gradient does not survive one iteration, so no tweak "
                  "rescues skipping renders. Stop here.")
        lines.append("")
        lines.append("    Magnitude matters separately: an optimiser reads a "
                     "scaled gradient as a changed learning rate, not as noise.")
        return "\n".join(lines)
