"""CPU-only state-machine checks for the Gaussian-SLAM active-stop guard."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.closed_loop_stop_guard import AuditedStopGuard


CFG = {
    "enabled": True,
    "calibration_frames": 4,
    "audit_every_proposals": 3,
    "history_size": 8,
    "min_history": 4,
    "audit_min_samples": 2,
}


guard = AuditedStopGuard(CFG)
for _ in range(4):
    guard.reset_frame()
    assert guard.decide_proposal(0.8, 1.0) == "audit"
    guard.end_frame(
        full_budget=True, coverage=0.8, innovation=1.0,
        audit_motion=0.01, audit_loss=0.001,
    )
assert len(guard._coverage_history) == 4
print("  PASS  calibration proposals retain full tails and build health history")


calibration_only = AuditedStopGuard(dict(CFG, audit_every_proposals=0))
for _ in range(4):
    calibration_only.reset_frame()
    assert calibration_only.decide_proposal(0.8, 1.0) == "audit"
    calibration_only.end_frame(
        full_budget=True, coverage=0.8, innovation=1.0,
        audit_motion=0.9, audit_loss=0.9,
    )
assert not calibration_only.latched_full_budget
calibration_only.reset_frame()
assert calibration_only.decide_proposal(0.01, 100.0) == "stop"
calibration_only.end_frame(full_budget=False, coverage=0.01, innovation=100.0)
assert not calibration_only.should_recover_initial(0.8, 0.01)
assert not calibration_only.latched_full_budget
assert calibration_only.periodic_audits == 0
print("  PASS  audit_every=0 keeps calibration tails but disables later forcing")


actions = []
for _ in range(3):
    guard.reset_frame()
    action = guard.decide_proposal(0.8, 1.0)
    actions.append(action)
    guard.end_frame(
        full_budget=(action != "stop"), coverage=0.8, innovation=1.0,
        audit_motion=(0.01 if action == "audit" else None),
        audit_loss=(0.001 if action == "audit" else None),
    )
assert actions == ["stop", "stop", "audit"]
print("  PASS  every third eligible proposal is a counterfactual audit")


guard.reset_frame()
assert guard.decide_proposal(0.05, 1.0) == "veto"
guard.end_frame(full_budget=True, coverage=0.8, innovation=1.0)
assert guard.coverage_vetoes == 1
print("  PASS  robust coverage collapse vetoes active stopping")


guard.reset_frame()
assert guard.decide_proposal(0.8, 100.0) == "veto"
guard.end_frame(full_budget=True, coverage=0.8, innovation=1.0)
assert guard.motion_vetoes == 1
print("  PASS  robust motion-prior innovation vetoes active stopping")


# Advance to the next periodic audit and make its full tail unsafe. Hard maxima
# latch immediately, independently of the p90 sample floor.
while True:
    guard.reset_frame()
    action = guard.decide_proposal(0.8, 1.0)
    if action == "audit":
        guard.end_frame(
            full_budget=True, coverage=0.8, innovation=1.0,
            audit_motion=0.9, audit_loss=0.001,
        )
        break
    guard.end_frame(full_budget=False, coverage=0.8, innovation=1.0)
assert guard.latched_full_budget
guard.reset_frame()
assert guard.decide_proposal(0.8, 1.0) == "audit"
print("  PASS  unsafe audit permanently falls back to full-budget tracking")


recovery = AuditedStopGuard(CFG)
for _ in range(4):
    recovery.reset_frame()
    recovery.end_frame(full_budget=True, coverage=0.8, innovation=1.0)
assert recovery.should_recover_initial(0.8, 0.05)
assert recovery.latched_full_budget and recovery.recoveries == 1
print("  PASS  optimizer-induced critical coverage collapse recovers prediction")


# --- release: off by default -----------------------------------------------
# The four checks above already prove release_after=0 (unset in CFG) leaves
# the latch permanent - "PASS unsafe audit permanently falls back" ran with
# no release config at all. This section is release_after > 0 specifically.

REL_CFG = dict(CFG, release_after=2)


def _latch(guard):
    """Drive one guard cleanly through calibration (so _coverage_history
    reaches min_history - should_recover_initial() needs that later), then
    trigger an unsafe periodic audit and leave it latched."""
    for _ in range(CFG["calibration_frames"]):
        guard.reset_frame()
        assert guard.decide_proposal(0.8, 1.0) == "audit"
        guard.end_frame(full_budget=True, coverage=0.8, innovation=1.0,
                        audit_motion=0.01, audit_loss=0.001)
    while True:
        guard.reset_frame()
        action = guard.decide_proposal(0.8, 1.0)
        if action == "audit":
            guard.end_frame(full_budget=True, coverage=0.8, innovation=1.0,
                            audit_motion=0.9, audit_loss=0.001)
            break
        guard.end_frame(full_budget=False, coverage=0.8, innovation=1.0)
    assert guard.latched_full_budget


def _latched_audit(guard, motion, loss):
    """One more latched-audit frame with a chosen (motion, loss) sample."""
    guard.reset_frame()
    action = guard.decide_proposal(0.8, 1.0)
    assert action == "audit", "still latched, every proposal must audit"
    guard.end_frame(full_budget=True, coverage=0.8, innovation=1.0,
                    audit_motion=motion, audit_loss=loss)


g = AuditedStopGuard(REL_CFG)
_latch(g)
_latched_audit(g, 0.01, 0.001)   # clean 1/2
assert g.latched_full_budget and not g.releases
_latched_audit(g, 0.01, 0.001)   # clean 2/2 -> release
assert not g.latched_full_budget and g.releases == 1
print("  PASS  release_after consecutive clean latched audits release the latch")


g = AuditedStopGuard(REL_CFG)
_latch(g)
_latched_audit(g, 0.01, 0.001)   # clean 1/2
_latched_audit(g, 0.9, 0.001)    # a bad sample resets the streak
assert g.latched_full_budget and not g.releases
_latched_audit(g, 0.01, 0.001)   # clean 1/2 again (streak restarted)
assert g.latched_full_budget and not g.releases
_latched_audit(g, 0.01, 0.001)   # clean 2/2 -> release
assert not g.latched_full_budget and g.releases == 1
print("  PASS  one bad sample resets the clean streak instead of releasing early")


g = AuditedStopGuard(REL_CFG)
_latch(g)
_latched_audit(g, 0.01, 0.001)   # clean 1/2 - would release next
assert g.should_recover_initial(0.8, 0.05)   # a fresh collapse (the arm
                                              # above was an unsafe AUDIT,
                                              # not a should_recover_initial
                                              # call, so this is recovery #1)
assert g.latched_full_budget and g.recoveries == 1
_latched_audit(g, 0.01, 0.001)   # 1/2 again, not 2/2 - the streak was voided
assert g.latched_full_budget and not g.releases
_latched_audit(g, 0.01, 0.001)   # now 2/2
assert not g.latched_full_budget and g.releases == 1
print("  PASS  a fresh collapse during probation voids the clean streak")


g = AuditedStopGuard(CFG)   # release_after=0 (unset): the class default
_latch(g)
for _ in range(10):
    _latched_audit(g, 0.01, 0.001)   # ten clean audits in a row
assert g.latched_full_budget and g.releases == 0
print("  PASS  release_after=0 never releases, no matter how many clean audits")


print("All closed-loop stop guard checks passed.")
