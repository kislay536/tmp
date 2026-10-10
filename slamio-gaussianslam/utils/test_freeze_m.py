"""accumulate_m: freezing MOMENTUM alone must not collapse into no_accum.

WHY THIS EXISTS. On a gradient-reuse iteration the same gradient is applied a
second time, and there are three things you can do with the optimizer state:

  accumulate both (default)   the step stops decaying - measured as it40+
                              0.06x against a baseline 0.03x on GSLAM/fr1_desk,
                              in a band where nothing reuses

  freeze both (no_accum)      catastrophic on SplaTAM: ATE 3.5 -> 10.8,
                              progress_rejects 0 -> 43

  freeze m only (freeze_m)    the arm under test

THE STATED MECHANISM FOR no_accum HAS BEEN WRONG TWICE in this repo, most
recently as "the state is identical so it takes the same step twice, doubling
the effective step". It does not: the second step still differs by ~2.5%, and
over a realistic schedule no_accum ends up taking LARGER polish-band steps than
accumulating does. The assertions below are what is actually measured, which is
the only reason this file exists rather than a comment.

The non-obvious part: P is only rebuilt by refactor(), so freezing M inside
step() cannot change THAT step. no_accum and freeze_m are byte-identical on the
reuse iteration itself and diverge only at the next refactor - which is why a
short probe makes the arm look like a no-op.

Runs on CPU.

    python utils/test_freeze_m.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pose_preconditioner import PosePreconditioner  # noqa: E402

DT = torch.float64
KW = dict(dim=6, lr=0.05, tau=0.05, refactor_every=1, carry=False, dtype=DT)

_fails = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          "" if not detail else "  (%s)" % detail))
    if not ok:
        _fails.append(name)


def _warmed(n=6):
    """A preconditioner with some history, so m and M are not at their init."""
    pre = PosePreconditioner(**KW)
    torch.manual_seed(0)
    for _ in range(n):
        pre.step(torch.randn(6, dtype=DT))
        pre.refactor(force=True)
    return pre


def _two_steps(**kw):
    """Apply the SAME gradient twice; return both deltas."""
    pre = _warmed()
    g = torch.full((6,), 0.3, dtype=DT)
    first = pre.step(g).clone()
    pre.refactor(force=True)
    second = pre.step(g, **kw).clone()
    return first, second


# --- the surprise: on the reuse iteration itself they are the SAME ---------
#
# P is only rebuilt by refactor(), so whether M advanced INSIDE step() cannot
# affect that same step. no_accum and freeze_m therefore return identical
# vectors on the reuse iteration; they differ only from the next refactor on.
# Worth pinning, because it is the reason a two-step probe says the arm does
# nothing when it plainly does.
_, b_noacc = _two_steps(accumulate=False)
_, b_freeze = _two_steps(accumulate=True, accumulate_m=False)
check("on the reuse iteration itself, no_accum == freeze_m (M is deferred)",
      torch.equal(b_noacc, b_freeze))

_, b_default = _two_steps()
check("both differ from accumulating m",
      not torch.allclose(b_default, b_freeze, atol=1e-9),
      "max diff = %.3e" % (b_default - b_freeze).abs().max())

check("freezing m takes a smaller step than accumulating it",
      b_freeze.norm() < b_default.norm(),
      "|freeze| %.6f vs |default| %.6f" % (b_freeze.norm(), b_default.norm()))


# --- and over a real refactor schedule they DO diverge ---------------------
# GSLAM's shape: refactor_every=10, reuse every other iteration after warmup.
def _bands(mode, n=60):
    pre = PosePreconditioner(dim=6, lr=0.01, tau=0.05, refactor_every=10,
                             carry=False, dtype=DT)
    torch.manual_seed(1)
    x = torch.randn(6, dtype=DT) * 2.0
    H = torch.diag(torch.tensor([4., 3., 2., 1., .5, .25], dtype=DT))
    out, prev_g = [], None
    for i in range(n):
        fresh = (i % 2 == 0) or i < 6
        g = H @ x if fresh else prev_g
        if fresh:
            prev_g = g.clone()
        kw = {}
        if not fresh and mode == "no_accum":
            kw = dict(accumulate=False)
        elif not fresh and mode == "freeze_m":
            kw = dict(accumulate=True, accumulate_m=False)
        d = pre.step(g, **kw)
        pre.refactor()
        x = x + d
        out.append(d.norm().item())
    return out


_d, _n, _f = _bands("default"), _bands("no_accum"), _bands("freeze_m")
_polish = lambda v: sum(v[30:]) / len(v[30:])
check("no_accum and freeze_m diverge once refactor rebuilds P",
      max(abs(a - b) for a, b in zip(_n, _f)) > 1e-4,
      "max per-step diff = %.3e" % max(abs(a - b) for a, b in zip(_n, _f)))

check("freeze_m gives the SMALLEST polish-band step of the three",
      _polish(_f) < _polish(_d) and _polish(_f) < _polish(_n),
      "freeze %.3e | default %.3e | no_accum %.3e"
      % (_polish(_f), _polish(_d), _polish(_n)))


# --- M bookkeeping ---------------------------------------------------------
pre = _warmed()
g = torch.full((6,), 0.3, dtype=DT)
M_before = pre.M.clone()
pre.step(g, accumulate=True, accumulate_m=False)
check("freeze_m still advances M",
      not torch.allclose(M_before, pre.M, atol=1e-12),
      "max|dM| = %.3e" % (M_before - pre.M).abs().max())

pre = _warmed()
M_before = pre.M.clone()
pre.step(g, accumulate=False)
check("no_accum freezes M too", torch.allclose(M_before, pre.M, atol=1e-12))

pre_a, pre_b = _warmed(), _warmed()
g2 = torch.full((6,), -0.17, dtype=DT)
check("omitting accumulate_m reproduces the old behaviour exactly",
      torch.equal(pre_a.step(g2), pre_b.step(g2, accumulate_m=None)))

print("all passed" if not _fails else "FAILED: %s" % ", ".join(_fails))
sys.exit(1 if _fails else 0)
