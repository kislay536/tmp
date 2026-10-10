"""GradStalenessProbe: every consecutive pair, not just every other one.

WHY THIS EXISTS. observe() used to check `iter_idx in self.starts` FIRST and
return immediately, so an iteration that was BOTH a configured start AND the
offset=1 close of the PREVIOUS window silently dropped that close - back-to-
back integer starts (needed for a dense g1-g2, g2-g3, g3-g4, ... trace, not
just every-other-pair) lost most of their measurements to this. Every dense
config in this project (splatam_gradstale_adam_dense.py, GRADSTALE_DENSE in
splatam_precond.py) worked around it with odd-only spacing, at half density.

THE FIX closes the active window with the freshly computed gradient BEFORE
checking whether this same iteration should also open a new one. This test
locks in three properties a short manual test would not catch:

  1. starts=[1,2,3,4,5], offsets=[1] records ALL FOUR consecutive pairs
     (1,2), (2,3), (3,4), (4,5) - the property this fix exists for.
  2. The original sparse go/no-go shape (starts=[5,15], offsets=[1,2,4],
     non-adjacent starts) is completely unaffected - the close-then-open
     reordering is a no-op when starts and closes never coincide.
  3. A GENUINE misconfiguration (a start requested before the ACTIVE
     window's own offsets finish) still warns and still drops the
     incomplete window, rather than the reordering silently hiding it.

Runs on CPU - everything here is arithmetic on small tensors, no CUDA needed.

    python utils/test_grad_staleness.py
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.grad_staleness import GradStalenessProbe  # noqa: E402

_fails = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          "" if not detail else "  (%s)" % detail))
    if not ok:
        _fails.append(name)


_cur = {"g": None}


def _grad_map():
    return _cur["g"]


def _feed(probe, frame, iters_and_grads):
    """Drive the probe through a frame exactly as a real caller would:
    wants() gates whether observe() is even called."""
    probe.reset_frame()
    for it, g in iters_and_grads:
        if probe.wants(frame, it):
            _cur["g"] = g
            probe.observe(frame, it, [], lambda _m: None, grad_map_fn=_grad_map)


# -- 1. every consecutive pair, not every other ------------------------------
g1 = torch.tensor([1.0, 0.0])
g2 = torch.tensor([1.0, 0.0])   # cos(g1,g2) = 1
g3 = torch.tensor([0.0, 1.0])   # cos(g2,g3) = 0
g4 = torch.tensor([0.0, -1.0])  # cos(g3,g4) = -1
g5 = torch.tensor([0.0, -1.0])  # cos(g4,g5) = 1

probe1 = GradStalenessProbe(dict(enabled=True, frames=[0],
                                 starts=list(range(1, 6)), offsets=[1],
                                 out_path=""))
_feed(probe1, 0, [(1, g1), (2, g2), (3, g3), (4, g4), (5, g5)])
check("every consecutive pair recorded (4 of 4)", len(probe1.rows) == 4,
      "rows=%d" % len(probe1.rows))
if len(probe1.rows) == 4:
    cos_vals = [r["cos_stale"] for r in probe1.rows]
    starts_seen = [r["start"] for r in probe1.rows]
    check("pairs in order (1,2,3,4)", starts_seen == [1, 2, 3, 4],
          "starts=%s" % starts_seen)
    check("cos values exact", all(abs(c - e) < 1e-9 for c, e in
                                  zip(cos_vals, [1.0, 0.0, -1.0, 1.0])),
          "cos=%s" % cos_vals)

# -- 2. the original sparse shape is unaffected ------------------------------
sparse_grads = {i: torch.tensor([1.0, float(i) * 0.01]) for i in range(1, 21)}
probe2 = GradStalenessProbe(dict(enabled=True, frames=[0], starts=[5, 15],
                                 offsets=[1, 2, 4], out_path=""))
_feed(probe2, 0, [(i, sparse_grads[i]) for i in range(1, 21)])
check("sparse shape: 2 starts x 3 offsets = 6 rows", len(probe2.rows) == 6,
      "rows=%d" % len(probe2.rows))
if len(probe2.rows) == 6:
    seen = sorted((r["start"], r["offset"]) for r in probe2.rows)
    expect = sorted((s, o) for s in (5, 15) for o in (1, 2, 4))
    check("sparse shape: exact (start, offset) pairs", seen == expect,
          "seen=%s" % seen)

# -- 3. a genuine misconfiguration still warns and still drops the window ---
# starts=[1, 2] with offsets=[1, 2]: window opened at 1 needs iterations 2
# AND 3 to complete. Iteration 2 closes offset=1 (recorded) but the window
# is still open (offset=2 still pending) when 2 also appears in `starts` -
# a real overlap, not the close-then-reopen case fix #1 legitimises.
probe3 = GradStalenessProbe(dict(enabled=True, frames=[0], starts=[1, 2],
                                 offsets=[1, 2], out_path=""))
buf = io.StringIO()
with redirect_stdout(buf):
    _feed(probe3, 0, [(1, torch.tensor([1.0, 0.0])),
                      (2, torch.tensor([1.0, 0.0])),
                      (3, torch.tensor([1.0, 0.0]))])
printed = buf.getvalue()
check("genuine overlap still warns", "overlaps an open window" in printed,
      printed.strip() or "(nothing printed)")
check("genuine overlap: offset=2 of the FIRST window was dropped",
      not any(r["start"] == 1 and r["offset"] == 2 for r in probe3.rows),
      "rows=%s" % [(r["start"], r["offset"]) for r in probe3.rows])
check("genuine overlap: offset=1 of the SECOND window (start=2) still recorded",
      any(r["start"] == 2 and r["offset"] == 1 for r in probe3.rows),
      "rows=%s" % [(r["start"], r["offset"]) for r in probe3.rows])

print("all passed" if not _fails else "FAILED: %s" % ", ".join(_fails))
sys.exit(1 if _fails else 0)
