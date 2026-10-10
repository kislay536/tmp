"""Adam's realised SE(3) tangent step magnitude, measured the same way the
preconditioner measures its own.

WHY THIS EXISTS. The preconditioner prints `step |d| mean` on every run, and
that number turned out to be the most predictive quantity in the whole record:
achievable accuracy divided by the step floor calls success, failure, and - as
of ScanNet - insensitivity, across three models and three scenes.

    SplaTAM TUM      6.5x  works        SplaTAM Replica  1.93x  fails
    SplaTAM Replica  5.5x  works        GSLAM   Replica  0.89x  fails
    MonoGS  TUM      3.5x  works        SplaTAM Replica   0.9x  fails
    SplaTAM ScanNet  29-140x  insensitive across an 8x LR range

But the ratio needs ACHIEVABLE ACCURACY, and the only source for that so far
has been ATE - which needs ground truth, so it cannot be computed at runtime
and cannot be used to tune anything on a new sequence.

THE PROPOSED SUBSTITUTE, AND WHAT THIS MODULE IS FOR. At a noise-limited
plateau an optimiser random-walks: the gradient's mean has collapsed relative
to its RMS, and the residual error settles at the radius where the restoring
force balances the step. Error and step are then the SAME SIZE - which is
exactly the mechanism the ladder recorded from the other side, when Replica's
pinned 3.53e-3 step sat against a 3.0e-3 accuracy floor and the pose orbited
at the radius of the answer.

If that holds, ADAM'S OWN PLATEAU STEP IS A GROUND-TRUTH-FREE ESTIMATE OF THE
ACCURACY THE SCENE SUPPORTS. It is only a valid estimate because the model
authors already tuned Adam's rate for the scene; it measures what THAT
optimiser can resolve, not an intrinsic property of the geometry.

NOBODY HAS CHECKED IT. That is the point of this module. The check is
concrete: run the Adam control, read the plateau step, and compare it against
the Adam ATE we already have for that cell -

    SplaTAM Replica   Adam ATE 0.23 cm = 2.3e-3 m
    SplaTAM ScanNet   Adam ATE 4.63 cm = 4.63e-2 m

If the plateau step lands near those, the substitute works and the tuner needs
no ground truth. If it does not, the ratio stays a post-hoc diagnostic and the
tuner needs a different reference. EITHER ANSWER IS WORTH HAVING; this only
supplies the measurement.

WHY IT CANNOT REUSE THE PRECONDITIONER'S ACCUMULATOR. `note_applied_step` does
exactly this arithmetic, but it lives on PosePreconditioner and the arm that
needs measuring is the one where `pose_pre is None`. The handoff path's own
comment ("the measurement this phase never had") describes the same gap from
inside a run that does have the object.

NO PER-ITERATION SYNC. Everything accumulates in device tensors and is read
once, at a checkpoint or at the end. A `.item()` per tracking iteration would
put a host sync inside the loop being measured - which is how three earlier
probes in this repo perturbed the thing they were measuring.
"""

import torch

# Same bands the preconditioner's STEP PROFILE uses, so the two summaries can
# be read against each other without rescaling.
_BANDS = ((0, 5), (5, 10), (10, 20), (20, 10 ** 9))


class AdamStepProbe:
    """Accumulates |dxi| per tracking iteration, banded by within-frame index.

    `record` takes the SE(3) tangent delta produced by tangent_of_pose_delta -
    NOT a 7-parameter (q, t) difference. The normalisation inside that helper
    drops the quaternion gauge, which Adam moves freely and the loss cannot
    see; a raw (q, t) delta norm would be a different quantity and would not
    be comparable with the preconditioner's `step |d|`.
    """

    def __init__(self, device=None, dtype=torch.float32):
        kw = dict(device=device, dtype=dtype)
        self._sum = torch.zeros((), **kw)
        self._max = torch.zeros((), **kw)
        self._n = torch.zeros((), **kw)
        # Per-band sums, so the within-frame DECAY is measurable and not just
        # the mean. The decay is what distinguishes "converged inside the
        # budget" from "still moving when the budget ran out" - the failure
        # mode that produced 99.79 cm at a perfectly healthy mean.
        self._bsum = torch.zeros(len(_BANDS), **kw)
        self._bmax = torch.zeros(len(_BANDS), **kw)
        self._bn = torch.zeros(len(_BANDS), **kw)
        self.frames = 0

    def record(self, dxi, it):
        """One tracking iteration. `it` is the within-frame iteration index."""
        with torch.no_grad():
            d = dxi.norm()
            self._sum.add_(d)
            self._n.add_(1.0)
            torch.maximum(self._max, d, out=self._max)
            for b, (lo, hi) in enumerate(_BANDS):
                if lo <= it < hi:
                    self._bsum[b].add_(d)
                    self._bn[b].add_(1.0)
                    torch.maximum(self._bmax[b], d, out=self._bmax[b])
                    break

    def note_frame(self):
        self.frames += 1

    def stats(self):
        """One host read. Returns None before any step has been recorded."""
        n = float(self._n)
        if n <= 0:
            return None
        mean = float(self._sum) / n
        bands = []
        for b, (lo, hi) in enumerate(_BANDS):
            bn = float(self._bn[b])
            if bn <= 0:
                continue
            bands.append((lo, hi, float(self._bsum[b]) / bn,
                          float(self._bmax[b]), int(bn)))
        return dict(mean=mean, max=float(self._max), n=int(n),
                    frames=self.frames, bands=bands)

    def summary(self, prefix="Adam step probe"):
        s = self.stats()
        if s is None:
            return f"{prefix}: no steps recorded"
        # `late` is the last band's mean - the plateau estimate, and the
        # quantity the accuracy substitute above actually proposes. `mean` is
        # what the existing ratio table was built from, so both are printed:
        # they answer different questions and the record needs to keep them
        # apart.
        late = s["bands"][-1][2] if s["bands"] else float("nan")
        out = (f"{prefix}: frames={s['frames']}, steps={s['n']}, "
               f"step |d| mean {s['mean']:.2e} max {s['max']:.2e}, "
               f"late-band mean {late:.2e}")
        if s["bands"]:
            parts = []
            for lo, hi, bmean, bmax, bn in s["bands"]:
                tag = f"it{lo}-{hi - 1}" if hi < 10 ** 9 else f"it{lo}+"
                parts.append(f"{tag} mean {bmean:.2e} max {bmax:.2e} n={bn}")
            out += "\n  by iteration band:  " + " | ".join(parts)
        out += ("\n  ACCURACY SUBSTITUTE: compare 'late-band mean' against this "
                "cell's Adam ATE in metres.\n  If they agree the ratio can be "
                "computed at runtime with no ground truth; if not, it cannot.")
        return out
