"""precond_probe.py - does the SAMPLE-EFFICIENCY ARGUMENT actually hold?

WHY THIS EXISTS. Every measurement on the sample metric so far is ATE at a
FIXED iteration budget, and that is the one thing which cannot test the claim.
The claim is:

    the rank-1 EMA is a poor estimator of M, so it needs many iterations to
    learn a metric whose purpose is to remove iterations

ATE at a fixed budget reports the OUTCOME after the estimator has had all the
iterations it wanted. If the rank-1 estimator catches up by iteration 20 and
the budget is 40, the two arms MUST agree - and they did, at i40, i30 and i20.
That is not evidence against the mechanism; it is evidence that a saturated
budget cannot see it.

THE THREE QUANTITIES THAT TEST IT DIRECTLY, none of which is ATE:

  1. CATCH-UP TIME.  cos(M_rank1_ema, M_sample_ema) as a function of the
     WITHIN-FRAME iteration index. This is the whole argument as a number: if
     the rank-1 EMA agrees with the well-estimated metric by iteration k, the
     sample metric can only buy something at budgets below k, and k is
     measurable rather than argued. Both EMAs run side by side on the SAME
     gradient stream, so the comparison is exact and carries no run-to-run
     noise at all - it is not two runs to be differenced, it is two functions
     of identical numbers.

  2. RANK AND CONDITIONING.  The rank-1 EMA cannot be full rank before
     iteration 6, so early steps go through a singular metric held up by the
     Levenberg floor. The numerical rank and the condition number say for how
     many iterations the FLOOR, not the metric, is doing the work.

  3. CONVERGENCE PER RENDER.  |g|/|g_0| against iteration index. This is the
     quantity the method claims to improve, and it is budget-independent: it
     needs no stopping rule switched on, and it does not care what ATE the
     frame eventually reaches.

THIS IS A DIAGNOSTIC RUN, NEVER A TIMED ONE. Every row costs a 6x6 eigh and
several host syncs inside the tracking loop. Do not quote wall time, ms/iter or
in-loop from a run with the probe enabled - the same rule the gradient-variance
and GN probes already carry.
"""

from __future__ import annotations

import json
import os

import torch


def _cos(A: torch.Tensor, B: torch.Tensor) -> float:
    """Frobenius alignment of two matrices, scale-free in both."""
    na = float(A.norm())
    nb = float(B.norm())
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float((A * B).sum()) / (na * nb)


def _spec(M: torch.Tensor, tol: float = 1e-10):
    """(condition number, numerical rank) of a symmetric PSD matrix."""
    ev = torch.linalg.eigvalsh(M.double())
    lo = float(ev.min())
    hi = float(ev.max())
    if hi <= 0.0:
        return float("inf"), 0
    rank = int((ev > tol * hi).sum())
    return (hi / lo if lo > 0 else float("inf")), rank


class PrecondProbe:
    """Side-by-side estimator comparison on one shared gradient stream.

    The sample arm supplies BOTH `g` and `M_inst` every iteration, so the
    rank-1 estimator can be run as a shadow for the cost of one `addr_` on a
    6x6. That is what makes the catch-up curve exact.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        # Frames are SAMPLED, not all recorded. A row per iteration per frame
        # over 592 frames is a large file for no extra information: the curve
        # is a within-frame shape and frames are replicates of it.
        self.every = int(cfg.get("every", 20))
        self.beta2 = float(cfg.get("beta2", 0.95))
        self.path = str(cfg.get("path", "./precond_probe.jsonl"))
        self.rows = []
        self._shadow = None
        self._frame = -1
        self._it = 0
        print(f"[PrecondProbe] constructed (enabled={self.enabled}, "
              f"every={self.every} frames, path={self.path})", flush=True)

    # -- lifecycle ---------------------------------------------------------

    def start_frame(self, frame: int, M_carried: torch.Tensor) -> None:
        """Called once per frame, AFTER carry_frame.

        The shadow starts from the SAME carried metric the real optimiser
        starts from, so the curve measures two estimators diverging from a
        common initial condition rather than from an arbitrary one.
        """
        self._frame = int(frame)
        self._it = 0
        if self.active:
            self._shadow = M_carried.detach().clone()

    @property
    def active(self) -> bool:
        return (self.enabled and self._frame >= 0
                and self._frame % self.every == 0)

    def record(self, g: torch.Tensor, M_inst: torch.Tensor,
               M_real: torch.Tensor, rel: torch.Tensor) -> None:
        """One iteration. `M_real` is the live metric the step actually used."""
        if not self.active or self._shadow is None:
            return
        # The rank-1 estimator, fed the same gradient, with the same horizon.
        # Meaningless when M_inst IS M (no sample metric to compare against) -
        # the cos_sample_rank1 column is then self-referential and must be
        # ignored. The rel_grad / rank / conditioning columns stay valid, and
        # those are the ones a STOPPING question asks for.
        self._shadow.mul_(self.beta2).addr_(g, g, alpha=1.0 - self.beta2)
        cond_r, rank_r = _spec(M_real)
        cond_s, rank_s = _spec(self._shadow)
        ggT = torch.outer(g, g)
        self.rows.append({
            "frame": self._frame,
            "it": self._it,
            # THE CATCH-UP CURVE. 1.0 means the rank-1 EMA has become the
            # sample metric and there is nothing left to buy at this budget.
            "cos_sample_rank1": _cos(M_real, self._shadow),
            # How much of each metric is still one direction.
            "cos_rank1_ggT": _cos(self._shadow, ggT),
            "cos_sample_ggT": _cos(M_real, ggT),
            "cos_inst_ggT": _cos(M_inst, ggT),
            "rank_sample": rank_r,
            "rank_rank1": rank_s,
            "cond_sample": cond_r,
            "cond_rank1": cond_s,
            "rel_grad": float(rel),
            "g_norm": float(g.norm()),
        })
        self._it += 1

    # -- output ------------------------------------------------------------

    def save(self) -> None:
        if not self.enabled or not self.rows:
            return
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.path, "w") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")
        print(f"[PrecondProbe] wrote {len(self.rows)} rows to {self.path}",
              flush=True)

    def summary(self) -> str:
        if not self.enabled:
            return "Preconditioner probe: disabled"
        if not self.rows:
            return "Preconditioner probe: enabled but NO ROWS - wiring is broken"
        # THE CURVE AT FIXED CHECKPOINTS, NOT A THRESHOLD CROSSING.
        #
        # A threshold was the first design and it was wrong: the rank-1 EMA
        # PLATEAUS below 1 rather than converging to it, because beta2=0.95
        # gives it ~20 effective samples for 21 parameters, so irreducible
        # estimation noise remains no matter how long the frame runs. On a
        # synthetic stationary stream it sat at 0.95 through iteration 59 and
        # "reaches cos>0.99" was therefore never true - a number that cannot
        # fire is not a measurement.
        #
        # The plateau VALUE is itself the answer to "does the rank-1 estimator
        # ever become the well-estimated metric": no, and this says by how much
        # it misses. The early points say how long it is singular, which is how
        # long the Levenberg floor is doing the steering rather than the metric.
        by_frame = {}
        for r in self.rows:
            by_frame.setdefault(r["frame"], []).append(r)
        for rows in by_frame.values():
            rows.sort(key=lambda r: r["it"])

        def at(it, key):
            vals = [rows[it][key] for rows in by_frame.values()
                    if len(rows) > it]
            return sum(vals) / len(vals) if vals else float("nan")

        pts = [i for i in (0, 1, 2, 5, 10, 20, 30, 40)
               if any(len(r) > i for r in by_frame.values())]
        curve = "  ".join(f"it{i}={at(i, 'cos_sample_rank1'):.3f}" for i in pts)
        # How many iterations the rank-1 metric spends rank-deficient, which is
        # how many steps are taken through a metric the floor is holding up.
        sing = []
        for rows in by_frame.values():
            k = next((r["it"] for r in rows
                      if r["rank_rank1"] >= r["rank_sample"]), None)
            if k is not None:
                sing.append(k)
        nl = chr(10)
        return ("Preconditioner probe: "
                f"{len(self.rows)} rows over {len(by_frame)} frames" + nl
                + f"  cos(M_sample, M_rank1) vs within-frame iter:  {curve}" + nl
                + (f"  rank-1 metric is rank-deficient for the first "
                   f"{sum(sing) / len(sing):.1f} iters "
                   f"({len(sing)}/{len(by_frame)} frames reached full rank)"
                   if sing else
                   "  rank-1 metric NEVER reached the sample metric's rank")
                + nl + "  rel_grad at those iters:  "
                + "  ".join(f"it{i}={at(i, 'rel_grad'):.3f}" for i in pts))
