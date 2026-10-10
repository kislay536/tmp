"""The render clock: a windowed stopper that only sees rendered iterations.

WHY THIS EXISTS. Under gradient reuse the windowed stopper counts every
iteration, reused ones included. A reused iteration carries the previous loss
and a stale-gradient step, so it dilutes the evidence and pushes stops later
(measured on MonoGS fr1: per render the two arms match, per iteration they do
not). RenderClock renumbers iterations so only rendered ones advance the
stopper's clock. The properties that would break silently:

  1. With reused steps of zero, feeding an interleaved sequence through the
     clock must give EXACTLY the decisions of the render-only sequence - same
     stop point, same counters. Any off-by-one in the renumbering shows here.
  2. A reused row must never reach the rule, and its step must not be lost:
     it is banked and lands in the next rendered row.
  3. `before(it)` is the phase-start coordinate on this clock.

Runs on CPU.

    python utils/test_render_clock.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.grad_reuse import RenderClock  # noqa: E402
from utils.windowed_convergence import IncumbentEnergyConvergence  # noqa: E402

CFG = {
    "enabled": True, "shadow": False, "kind": "incumbent_energy",
    "patience": 1, "check_every": 2, "pose_change": 0.1,
    "loss_change": 0.001, "progress_min": 0.0, "proposal_after_phase": 0,
    "energy_window": 4, "energy_patience": 2, "decay_ratio": 0.2,
}


def fresh_sequence(n):
    """Amplitude 1.0 then 0.1, alternating sign, flat loss - the shape the
    existing windowed-convergence test fires on."""
    out = []
    for i in range(n):
        amp = 1.0 if i < 4 else 0.1
        out.append((1.0, np.array([amp * ((-1.0) ** i), 0, 0, 0, 0, 0])))
    return out


def run_plain(seq):
    rule = IncumbentEnergyConvergence(CFG, scales=[1.0] * 6)
    rule.reset_frame()
    fired = []
    for i, (loss, step) in enumerate(seq):
        if rule.observe(i, loss, step):
            fired.append(i + 1)
    rule.end_frame()
    return fired, rule


def run_clocked(seq, reused_mask, reused_step):
    """seq: the rendered rows; a reused row (zero or given step, stale loss) is
    inserted wherever reused_mask says, in iteration order."""
    clock = RenderClock()
    rule = IncumbentEnergyConvergence(CFG, scales=[1.0] * 6)
    rule.reset_frame()
    fired, k, it, last_loss = [], 0, 0, 1.0
    calls = []
    while k < len(seq):
        reused = bool(reused_mask(it)) and k > 0
        clock.note(reused)
        if reused:
            loss, step = last_loss, np.asarray(reused_step, dtype=np.float64)
        else:
            loss, step = seq[k]
            k += 1
            last_loss = loss
        mapped = clock.map_row(it, step)
        if mapped is not None:
            idx, s = mapped
            calls.append((idx, s.copy()))
            if rule.observe(idx, loss, s):
                fired.append(idx + 1)
        it += 1
    rule.end_frame()
    return fired, rule, calls


def test_equivalence_with_zero_reused_steps():
    seq = fresh_sequence(16)
    base_fired, base_rule = run_plain(seq)
    assert base_fired, "the reference sequence never fires: bad fixture"
    # Reuse every other iteration after the first few, zero-step.
    fired, rule, calls = run_clocked(
        seq, lambda it: it >= 3 and it % 2 == 1, [0, 0, 0, 0, 0, 0])
    assert fired == base_fired, (fired, base_fired)
    assert rule.energy_rejects == base_rule.energy_rejects
    assert rule.energy_patience_waits == base_rule.energy_patience_waits
    # The rule saw render indices 0..N-1 exactly once each, in order.
    assert [c[0] for c in calls] == list(range(len(calls))), [c[0] for c in calls]
    print("  PASS  interleaved reused rows change nothing when their steps are zero")


def test_reused_steps_are_banked_into_the_next_rendered_row():
    clock = RenderClock()
    for r in [0, 0, 1, 0, 1, 1, 0]:
        clock.note(bool(r))
    got = [clock.map_row(i, np.full(6, float(i + 1))) for i in range(7)]
    assert [g is None for g in got] == [False, False, True, False, True, True, False]
    assert got[3][0] == 2 and np.allclose(got[3][1], 4 + 3), got[3]
    assert got[6][0] == 3 and np.allclose(got[6][1], 7 + 5 + 6), got[6]
    print("  PASS  a reused step is banked and lands in the next rendered row")


def test_fresh_evidence_drops_reused_steps():
    clock = RenderClock(bank_steps=False)
    for r in [0, 0, 1, 0, 1, 1, 0]:
        clock.note(bool(r))
    got = [clock.map_row(i, np.full(6, float(i + 1))) for i in range(7)]
    assert [g is None for g in got] == [False, False, True, False,
                                        True, True, False]
    # The reused rows 2, 4 and 5 never reach the rule and, unlike the
    # conservative render clock, are not added to rows 3 and 6.
    assert got[3][0] == 2 and np.allclose(got[3][1], 4), got[3]
    assert got[6][0] == 3 and np.allclose(got[6][1], 7), got[6]
    print("  PASS  fresh-evidence mode drops reused steps instead of banking")


def test_trailing_reused_rows_reach_nothing():
    clock = RenderClock()
    for r in [0, 1, 1]:
        clock.note(bool(r))
    assert clock.map_row(0, np.ones(6)) is not None
    assert clock.map_row(1, np.ones(6)) is None
    assert clock.map_row(2, np.ones(6)) is None
    print("  PASS  reused rows at the end of a frame produce no observation")


def test_before_is_the_render_count_so_far():
    clock = RenderClock()
    for r in [0, 0, 1, 1, 0]:
        clock.note(bool(r))
    assert [clock.before(i) for i in range(5)] == [0, 1, 2, 2, 2], \
        [clock.before(i) for i in range(5)]
    print("  PASS  before(it) is the number of renders completed before it")


def test_a_naive_iteration_axis_is_not_equivalent():
    """Documents WHY the clock exists: feed the same interleaved rows straight
    to the rule and the decisions differ from the render-only sequence."""
    seq = fresh_sequence(16)
    base_fired, _ = run_plain(seq)
    rule = IncumbentEnergyConvergence(CFG, scales=[1.0] * 6)
    rule.reset_frame()
    fired, k, it = [], 0, 0
    while k < len(seq):
        if it >= 3 and it % 2 == 1:
            loss, step = 1.0, np.zeros(6)
        else:
            loss, step = seq[k]
            k += 1
        if rule.observe(it, loss, step):
            fired.append(it + 1)
        it += 1
    rule.end_frame()
    assert fired != base_fired, (fired, base_fired)
    print("  PASS  without the clock the same rows give a different stop "
          "(%s vs %s)" % (fired, base_fired))


if __name__ == "__main__":
    print("render clock")
    test_equivalence_with_zero_reused_steps()
    test_reused_steps_are_banked_into_the_next_rendered_row()
    test_fresh_evidence_drops_reused_steps()
    test_trailing_reused_rows_reach_nothing()
    test_before_is_the_render_count_so_far()
    test_a_naive_iteration_axis_is_not_equivalent()
    print("all passed")
