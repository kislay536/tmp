import os
import sys
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.es_signals import ESSignalBuffer
from utils.pose_preconditioner import tangent_of_pose_delta
from utils.windowed_convergence import (
    AutoIncumbentConvergence,
    AutoIncumbentEnergyConvergence,
    AutoWindowedConvergence,
    IncumbentConvergence,
    IncumbentEnergyConvergence,
    WindowedConvergence,
    WindowedConvergenceSweep,
    _pose_distance_numpy,
    _se3_exp_numpy,
    make_windowed_convergence,
)


print("[0] incumbent-energy keeps a stable incumbent armed until energy decays")
combined_cfg = {
    "enabled": True,
    "shadow": True,
    "kind": "incumbent_energy",
    "patience": 1,
    "check_every": 2,
    "pose_change": 0.1,
    "loss_change": 0.001,
    "progress_min": 0.0,
    "proposal_after_phase": 0,
    "energy_window": 4,
    "energy_patience": 2,
    "decay_ratio": 0.2,
}
combined = IncumbentEnergyConvergence(combined_cfg, scales=[1.0] * 6)
combined.reset_frame()
combined_fired = []
for i in range(16):
    amplitude = 1.0 if i < 4 else 0.1
    step = [amplitude * ((-1.0) ** i), 0.0, 0.0, 0.0, 0.0, 0.0]
    if combined.observe(i, 1.0, step):
        combined_fired.append(i + 1)
combined.end_frame()
assert combined_fired == [10], combined_fired
assert combined.energy_rejects == 0 and combined.energy_patience_waits == 1
assert "Incumbent-energy convergence: shadow" in combined.summary()
assert "proposal decay med=0.1" in combined.summary()
print("  PASS  loss stability alone cannot bypass the normalized energy gate")

active_combined = IncumbentEnergyConvergence(
    dict(combined_cfg, shadow=False), scales=[1.0] * 6
)
active_combined.reset_frame()
active_fired = []
for i in range(12):
    amplitude = 1.0 if i < 4 else 0.1
    step = [amplitude * ((-1.0) ** i), 0.0, 0.0, 0.0, 0.0, 0.0]
    if active_combined.observe(i, 1.0, step):
        active_fired.append(i + 1)
        if len(active_fired) == 1:
            active_combined.note_barred_proposal()
active_combined.end_frame()
assert active_fired == [10, 12], active_fired
assert active_combined.barred_proposals == 1
assert active_combined.proposals == 1
assert active_combined.metrics(16)["mean_saved_per_frame"] == 4.0
print("  PASS  an active health veto re-arms the combined proposal")

print("[0a] repeated clean proposals can release only the anomaly veto")
hard_veto = IncumbentEnergyConvergence(
    dict(combined_cfg, shadow=False), scales=[1.0] * 6
)
hard_veto.reset_frame()
assert all(hard_veto.health_veto(True, False) for _ in range(5))
assert not hard_veto.health_veto(False, False)
assert hard_veto.health_veto(False, True)
hard_veto.end_frame()
assert hard_veto.anomaly_releases == 0
assert "anomaly_release_after=" not in hard_veto.summary()

released = IncumbentEnergyConvergence(
    dict(combined_cfg, shadow=False, anomaly_release_after=3),
    scales=[1.0] * 6,
)
released.reset_frame()
assert [released.health_veto(True, False) for _ in range(3)] == [
    True, True, False
]
assert released.anomaly_releases == 1
# Runtime drift is absolute even after the frozen start anomaly was released.
assert released.health_veto(True, True)
assert not released.health_veto(True, False)
released.end_frame()
released.reset_frame()
assert released.health_veto(True, False), "the evidence must reset each frame"
released.end_frame()
assert "anomaly_release_after=3, released=1" in released.summary()

released_stop = IncumbentEnergyConvergence(
    dict(combined_cfg, shadow=False, anomaly_release_after=3),
    scales=[1.0] * 6,
)
released_stop.reset_frame()
released_fired = []
for i in range(16):
    amplitude = 1.0 if i < 4 else 0.1
    step = [amplitude * ((-1.0) ** i), 0.0, 0.0, 0.0, 0.0, 0.0]
    if released_stop.observe(i, 1.0, step):
        if released_stop.health_veto(True, False):
            released_stop.note_barred_proposal()
        else:
            released_fired.append(i + 1)
            break
released_stop.end_frame()
assert released_fired == [14], released_fired
assert released_stop.barred_proposals == 2
assert released_stop.proposals == 1
assert released_stop.anomaly_releases == 1

try:
    IncumbentEnergyConvergence(
        dict(combined_cfg, anomaly_release_after=-1), scales=[1.0] * 6
    )
    raise AssertionError("negative anomaly release must be rejected")
except ValueError as exc:
    assert "anomaly_release_after" in str(exc)
try:
    make_windowed_convergence(
        dict(combined_cfg, kind="energy", anomaly_release_after=3),
        scales=[1.0] * 6,
    )
    raise AssertionError("energy-only rules cannot silently ignore release")
except ValueError as exc:
    assert "incumbent" in str(exc)
print("  PASS  default remains hard; release is per-frame; drift stays absolute")

print("[0b] frame-relative energy survives a low-energy phase transition")
phase_energy_cfg = dict(
    combined_cfg,
    proposal_after_phase=1,
    energy_patience=1,
    decay_ratio=0.1,
)
phase_energy_steps = [
    [amplitude * ((-1.0) ** i), 0.0, 0.0, 0.0, 0.0, 0.0]
    for i, amplitude in enumerate([1.0] * 4 + [0.4] * 4 + [0.08] * 4)
]

def phase_energy_fires(phase_relative):
    rule = IncumbentEnergyConvergence(
        dict(phase_energy_cfg, energy_phase_relative=phase_relative),
        scales=[1.0] * 6,
    )
    rule.reset_frame()
    fires = []
    for i, step in enumerate(phase_energy_steps):
        if i == 4:
            rule.start_phase(i, "diagonal")
        if rule.observe(i, 1.0, step):
            fires.append(i + 1)
    rule.end_frame()
    return rule, fires

phase_energy, phase_energy_hits = phase_energy_fires(True)
frame_energy, frame_energy_hits = phase_energy_fires(False)
assert phase_energy_hits == [], phase_energy_hits
assert frame_energy_hits == [12], frame_energy_hits
assert "reference=phase" in phase_energy.summary()
assert "reference=frame" in frame_energy.summary()
print("  PASS  frame reference remains attainable when a later phase starts low")


def run(cfg, steps, losses, scales=None):
    rule = WindowedConvergence(cfg, scales=scales or [1.0] * len(steps[0]))
    rule.reset_frame()
    fired = []
    for i, (step, loss) in enumerate(zip(steps, losses)):
        if rule.observe(i, loss, step):
            fired.append(i + 1)
    rule.end_frame()
    return rule, fired


CFG = {
    "enabled": True,
    "shadow": True,
    "window": 8,
    "check_every": 2,
    "patience": 2,
    "z_threshold": 2.0,
    "decay_ratio": 0.25,
}


print("[1] zero-mean pose motion and flat loss converge")
flat_steps = [
    [(1.0 if i < 8 else 0.1) * ((-1.0) ** i), 0.0]
    for i in range(40)
]
flat_losses = [1.0 + 1e-3 * ((-1.0) ** i) for i in range(40)]
rule, fired = run(CFG, flat_steps, flat_losses)
assert fired == [18], fired
assert rule.proposals == 1 and rule.frames == 1
assert rule.eligible_checks == 2 and rule.patience_waits == 1
assert rule.energy_rejects == rule.pose_rejects == rule.loss_rejects == 0
assert "gates eligible=2" in rule.summary()
assert "proposal bands n[post-motion/loss-gain]" in rule.summary(budget=40)
assert "post motion med/p90/p99/max=" in rule.summary(budget=40)
assert "post loss gain med/p90/p99/max=" in rule.summary(budget=40)
print("  PASS  waits for baseline, decay, and two complete evidence windows")


print("[2] coherent motion never masquerades as convergence")
coherent_steps = [
    [1.0, -2.0] if i < 8 else [1e-9, -2e-9]
    for i in range(40)
]
rule, fired = run(CFG, coherent_steps, [1.0] * 40)
assert fired == []
print("  PASS  arbitrarily small but consistently directed updates continue")


print("[3] either direction of significant loss trend continues")
zero = [[0.0, 0.0] for _ in range(40)]
for losses in ([1.0 - 0.01 * i for i in range(40)],
               [1.0 + 0.01 * i for i in range(40)]):
    _, fired = run(CFG, zero, losses)
    assert fired == []
print("  PASS  improving and worsening objectives are both non-converged")


print("[4] decisions are invariant to pose and loss units")
base_steps = [
    [(1.0 if i < 8 else 0.1) * ((-1.0) ** i),
     (0.25 if i < 8 else 0.025) * ((-1.0) ** (i + 1))]
    for i in range(40)
]
base_losses = [3.0 + 1e-3 * ((-1.0) ** i) for i in range(40)]
a, fired_a = run(CFG, base_steps, base_losses, scales=[0.5, 2.0])
scaled_steps = [[7.0 * x, 0.1 * y] for x, y in base_steps]
b, fired_b = run(CFG, scaled_steps,
                 [1000.0 * loss + 37.0 for loss in base_losses],
                 scales=[3.5, 0.2])
assert fired_a == fired_b
assert a.summary().split("; proposed iteration", 1)[1].split(", score", 1)[0] == \
       b.summary().split("; proposed iteration", 1)[1].split(", score", 1)[0]
print("  PASS  coordinate rates and affine loss units do not move the decision")


print("[5] invalid evidence cannot trigger convergence")
bad_steps = list(flat_steps)
bad_steps[7] = [np.nan, 0.0]
rule, fired = run(CFG, bad_steps, flat_losses)
assert rule.invalid == 1 and (not fired or fired[0] > 10)
print("  PASS  invalid rows reset the evidence window and streak")


print("[6] active/shadow is control wiring, not different arithmetic")
active_cfg = dict(CFG, shadow=False)
shadow, shadow_fired = run(CFG, flat_steps, [1.0] * 40)
active, active_fired = run(active_cfg, flat_steps, [1.0] * 40)
assert shadow_fired == active_fired == [18]
assert shadow.active is False and active.active is True
print("  PASS  both modes propose identically")


print("[7] a phase boundary discards mixed-regime evidence")
rule = WindowedConvergence(CFG, scales=[1.0, 1.0])
rule.reset_frame()
fired = []
for i in range(12):
    step = [((-1.0) ** i), 0.0]
    if rule.observe(i, 1.0, step):
        fired.append(i + 1)
rule.start_phase(12, "diagonal")
for i in range(12, 32):
    phase_i = i - 12
    amplitude = 1.0 if phase_i < 8 else 0.1
    step = [amplitude * ((-1.0) ** i), 0.0]
    if rule.observe(i, 1.0, step):
        fired.append(i + 1)
rule.end_frame()
assert fired == [30], fired
assert rule.phase_resets == 1
print("  PASS  the decision uses two complete windows from one phase only")


print("[8] a partial phase drain preserves chronological ring order")
signals = ESSignalBuffer(
    {"enabled": True, "batch": 8}, device="cpu", pose_dim=6
)
rot = torch.zeros(3)
tran = torch.zeros(3)


def record_signal(i):
    signals.record_pre_step(rot, tran)
    signals.record_post_step(
        torch.tensor(float(i)),
        torch.full((3,), float(i + 1)),
        torch.full((3,), float(i + 1)),
    )
    signals.note_iteration()


signals.reset_for_frame()
for i in range(4):
    record_signal(i)
first = signals.drain_phase()
for i in range(4, 12):
    record_signal(i)
second = signals.drain_if_full()
assert [row["iter"] for row in first] == list(range(4))
assert [row["iter"] for row in second] == list(range(4, 12))

# Adaptive proposal rejection has no honest loss/step pair for the rejected
# render. Native convergence still leaves valid earlier rows pending even when
# the windowed criterion is disabled, so they must be drained before the hole
# is advanced. This is the exact state that crashed the calibrated TUM run.
signals.reset_for_frame()
for i in range(3):
    record_signal(i)
try:
    signals.skip_iteration()
    raise AssertionError("skip accepted with native-convergence rows pending")
except RuntimeError:
    pass
native_prefix = signals.drain_phase()
signals.skip_iteration()
for i in range(4, 12):
    record_signal(i)
native_tail = signals.drain_if_full()
assert [row["iter"] for row in native_prefix] == [0, 1, 2]
assert [row["iter"] for row in native_tail] == list(range(4, 12))
assert [row["loss"] for row in second] == [float(i) for i in range(4, 12)]
print("  PASS  post-transition rows return oldest-first after a partial drain")


print("[9] a shadow sweep shares evidence but keeps independent thresholds")
sweep = WindowedConvergenceSweep(
    "loose_z:3:0.25,loose_decay:2:0.50", CFG, scales=[1.0, 1.0]
)
assert sweep.enabled and len(sweep.rules) == 2
sweep.reset_frame()
for i, (step, loss) in enumerate(zip(flat_steps, flat_losses)):
    sweep.observe(i, loss, step)
sweep.end_frame()
lines = sweep.summary_lines(budget=40)
assert len(lines) == 2
assert lines[0].startswith(
    "Windowed convergence sweep candidate=loose_z: shadow"
)
assert all("proposals=1/1" in line for line in lines)
try:
    WindowedConvergenceSweep("bad", CFG, scales=[1.0, 1.0])
except ValueError as exc:
    assert "label:z_threshold:decay_ratio" in str(exc)
else:
    raise AssertionError("malformed sweep specification was accepted")
try:
    WindowedConvergenceSweep(
        "unsafe:3:0.5", dict(CFG, shadow=False), scales=[1.0, 1.0]
    )
except ValueError as exc:
    assert "shadow-only" in str(exc)
else:
    raise AssertionError("active sweep was accepted")
print("  PASS  candidates are parsed, reported, and forced to remain shadow-only")


print("[10] z=inf isolates the dimensionless energy-decay gate")
energy_cfg = dict(CFG, z_threshold=float("inf"), decay_ratio=0.25)
rule, fired = run(energy_cfg, coherent_steps, [1.0] * len(coherent_steps))
assert fired == [18], fired
assert rule.energy_rejects == 0
assert rule.pose_rejects == rule.loss_rejects == 0
assert "z<=inf" in rule.summary()
active_rule, active_fired = run(
    dict(energy_cfg, shadow=False),
    coherent_steps,
    [1.0] * len(coherent_steps),
)
assert active_rule.active and active_fired == fired
print("  PASS  coherent but negligible updates are judged only by relative energy")


print("[11] SplaTAM signal rows preserve the exact pre/post pose pair")
pair_signals = ESSignalBuffer(
    {"enabled": True, "batch": 2, "pose_at_loss": True},
    device="cpu",
    pose_dim=7,
    extra_cols=1,
    include_pose_pair=True,
)
q0 = torch.tensor([2.0, 0.0, 0.0, 0.0])
t0 = torch.tensor([1.0, 2.0, 3.0])
q1 = torch.tensor([3.0, 0.0, 0.0, 0.0])
t1 = torch.tensor([1.1, 1.8, 3.3])
pair_signals.reset_for_frame()
pair_signals.record_pre_step(q0, t0)
pair_signals.record_post_step(
    torch.tensor(4.0), q1, t1, extra=[torch.tensor(7.0)]
)
pair_signals.note_iteration()
row = pair_signals.drain_remainder()[0]
assert torch.equal(row["rot"], q0)  # pose_at_loss candidate remains unchanged
assert torch.equal(row["pre_rot"], q0) and torch.equal(row["post_rot"], q1)
assert torch.allclose(row["pre_tran"], t0)
assert torch.allclose(row["post_tran"], t1)
assert row["extra"] == [7.0]
step = tangent_of_pose_delta(
    row["pre_rot"], row["pre_tran"], row["post_rot"], row["post_tran"]
)
assert torch.allclose(
    step, torch.tensor([0.1, -0.2, 0.3, 0.0, 0.0, 0.0], dtype=step.dtype),
    atol=1e-6,
)
print("  PASS  quaternion gauge is discarded and extra-column offsets survive")


print("[12] automatic calibration selects by measured full-tail regret")
auto_cfg = {
    "enabled": True,
    "shadow": False,
    "budget": 20,
    "auto_calibration_frames": 2,
    "auto_audit_every": 2,
    "auto_min_proposals": 2,
    "auto_loss_p90": 0.02,
    "auto_loss_max": 0.05,
    "auto_motion_p90": 0.25,
    "auto_motion_max": 0.75,
    "auto_spec": "fast:4:2:1:inf:0.25,slow:8:2:1:inf:0.25",
}
auto_steps = [
    [(1.0 if i < 4 else 0.1) * ((-1.0) ** i), 0.0]
    for i in range(20)
]
auto = AutoWindowedConvergence(auto_cfg, scales=[1.0, 1.0])
for _ in range(2):
    auto.reset_frame()
    assert not any(
        auto.observe(i, 1.0, step) for i, step in enumerate(auto_steps)
    )
    auto.end_frame()
assert "selected=fast" in auto.summary()
auto.reset_frame()
active_fires = [
    i + 1 for i, step in enumerate(auto_steps)
    if auto.observe(i, 1.0, step)
]
assert active_fires == [8], active_fires
auto.end_frame()
assert "runtime active" in auto.summary()
auto.note_barred_proposal()
assert "barred=1" in auto.summary()

# Frame four is the scheduled audit. A loss drop after fast's proposal but
# before slow's makes the selected rule unsafe while leaving the later rule
# safe, so the controller must tighten rather than continue blindly.
auto.reset_frame()
audit_losses = [1.0] * 12 + [0.8] * 8
assert not any(
    auto.observe(i, loss, step)
    for i, (step, loss) in enumerate(zip(auto_steps, audit_losses))
)
auto.end_frame()
assert "selected=slow" in auto.summary()

unsafe = AutoWindowedConvergence(
    dict(auto_cfg, auto_spec="fast:4:2:1:inf:0.25"),
    scales=[1.0, 1.0],
)
unsafe_losses = [1.0] * 8 + [0.8] * 12
for _ in range(2):
    unsafe.reset_frame()
    for i, (step, loss) in enumerate(zip(auto_steps, unsafe_losses)):
        assert not unsafe.observe(i, loss, step)
    unsafe.end_frame()
assert "selected=none" in unsafe.summary()
unsafe.reset_frame()
assert not any(
    unsafe.observe(i, loss, step)
    for i, (step, loss) in enumerate(zip(auto_steps, unsafe_losses))
)
unsafe.end_frame()
print("  PASS  safe fast rule activates, audit tightens, unsafe tail keeps full budget")

# A calibration skip is an exact full-budget no-op: it contributes no
# candidate evidence and selection happens only after the requested number of
# subsequent evidence frames.
skipped_cfg = dict(auto_cfg, auto_calibration_skip_frames=2)
skipped = AutoWindowedConvergence(skipped_cfg, scales=[1.0, 1.0])
for _ in range(2):
    skipped.reset_frame()
    assert not any(
        skipped.observe(i, 1.0, step) for i, step in enumerate(auto_steps)
    )
    skipped.end_frame()
assert skipped.calibration_seen == 0
assert "selected=none" in skipped.summary()
for _ in range(2):
    skipped.reset_frame()
    assert not any(
        skipped.observe(i, 1.0, step) for i, step in enumerate(auto_steps)
    )
    skipped.end_frame()
assert skipped.calibration_seen == 2
assert "selected=fast" in skipped.summary()
assert "skip=2" in skipped.summary()
print("  PASS  calibration skip excludes the unstable prefix from evidence")


print("[13] a phase-relative gate cannot skip the acquisition regime")
phase_cfg = dict(CFG, proposal_after_phase=1)
_, ungated_fires = run(phase_cfg, flat_steps, flat_losses)
assert ungated_fires == []
phase_rule = WindowedConvergence(phase_cfg, scales=[1.0, 1.0])
phase_rule.reset_frame()
phase_rule.start_phase(20, "diagonal")
phase_fires = [
    i + 21
    for i, (step, loss) in enumerate(zip(flat_steps, flat_losses))
    if phase_rule.observe(i + 20, loss, step)
]
phase_rule.end_frame()
assert phase_fires == [38], phase_fires
assert "phase>=1" in phase_rule.summary()
print("  PASS  no initial-phase proposal; identical evidence fires after transition")


print("[14] incumbent stability ignores motion after the best pose was found")
inc_cfg = {
    "enabled": True,
    "shadow": True,
    "patience": 4,
    "check_every": 2,
    "pose_change": 0.1,
    "loss_change": 0.001,
    "progress_min": 0.001,
    "proposal_after_phase": 1,
}
inc_steps = [[0.4, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(20)]
inc_losses = [10.0, 8.0, 6.0, 5.0, 4.0] + [4.0] * 15
inc = IncumbentConvergence(inc_cfg, scales=[1.0] * 6)
inc.reset_frame()
for i in range(4):
    assert not inc.observe(i, inc_losses[i], inc_steps[i])
inc.start_phase(4, "diagonal")
inc_fires = [
    i + 1 for i in range(4, 20)
    if inc.observe(i, inc_losses[i], inc_steps[i])
]
inc.end_frame()
inc.note_barred_proposal()
assert inc_fires == [10], inc_fires
assert inc.metrics(20)["post_motion"] == [0.0]
assert inc.metrics(20)["mean_saved_per_frame"] == 0.0
assert "incumbent regret" in inc.summary(20)
assert "barred=1" in inc.summary(20)

moving = IncumbentConvergence(
    dict(inc_cfg, proposal_after_phase=0), scales=[1.0] * 6
)
moving.reset_frame()
assert not any(
    moving.observe(i, 20.0 - i, step)
    for i, step in enumerate(inc_steps)
)
moving.end_frame()
assert moving.proposals == 0
print("  PASS  optimizer travel is irrelevant; continued incumbent progress is not")


print("[15] incumbent pose regret uses composed SE(3), not a step sum")
first = _se3_exp_numpy(np.asarray([0.3, -0.1, 0.2, 0.1, -0.2, 0.05]))
second = _se3_exp_numpy(np.asarray([-0.2, 0.4, 0.1, -0.05, 0.03, 0.2])) @ first
expected = np.linalg.norm(
    np.asarray([-0.2, 0.4, 0.1, -0.05, 0.03, 0.2])
) / np.sqrt(6.0)
assert abs(_pose_distance_numpy(first, second, np.ones(6)) - expected) < 1e-10
ordered = IncumbentConvergence(
    dict(inc_cfg, proposal_after_phase=0, step_order="rotation_translation"),
    scales=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
)
assert np.array_equal(ordered.scales, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
ordered.reset_frame()
ordered.observe(0, 1.0, [0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
assert np.allclose(
    ordered._current_pose,
    _se3_exp_numpy([0.4, -0.5, 0.6, 0.1, -0.2, 0.3]),
)
ordered.end_frame()
print("  PASS  non-commuting updates are compared through the exact relative pose")


print("[16] automatic incumbent calibration rewards safe early acquisition")
auto_inc_cfg = {
    "enabled": True,
    "shadow": False,
    "budget": 20,
    "auto_calibration_frames": 2,
    "auto_audit_every": 0,
    "auto_min_proposals": 2,
    "auto_loss_p90": 0.02,
    "auto_loss_max": 0.05,
    "auto_motion_p90": 0.25,
    "auto_motion_max": 0.75,
    "auto_min_saved_fraction": 0.25,
    "auto_spec": (
        "fast:4:2:0.1:0.001:0.001,"
        "slow:8:2:0.1:0.001:0.001"
    ),
}
auto_inc = AutoIncumbentConvergence(
    dict(auto_inc_cfg, auto_report_candidates=True), scales=[1.0] * 6
)
factory_inc = make_windowed_convergence(
    dict(auto_inc_cfg, auto_enabled=True, auto_kind="incumbent"),
    scales=[1.0] * 6,
)
assert isinstance(factory_inc, AutoIncumbentConvergence)
direct_factory_inc = make_windowed_convergence(
    dict(inc_cfg, kind="incumbent", shadow=False), scales=[1.0] * 6
)
assert isinstance(direct_factory_inc, IncumbentConvergence)
assert direct_factory_inc.active
frontier_output = StringIO()
with redirect_stdout(frontier_output):
    for _ in range(2):
        auto_inc.reset_frame()
        assert not any(
            auto_inc.observe(i, loss, step)
            for i, (loss, step) in enumerate(zip(inc_losses, inc_steps))
        )
        auto_inc.end_frame()
assert frontier_output.getvalue().count(
    "[AutoIncumbentConvergenceFrontier]\tBEGIN"
) == 1
assert "\tpredicted_iters\t" in frontier_output.getvalue()
assert "selected=fast" in auto_inc.summary()
frontier = auto_inc.candidate_frontier()
assert [row["candidate"] for row in frontier] == ["fast", "slow"]
assert frontier[0]["predicted_iters"] < frontier[1]["predicted_iters"]
assert frontier[0]["safe"] and frontier[0]["reasons"] == []
auto_inc.reset_frame()
active_fires = [
    i + 1 for i, (loss, step) in enumerate(zip(inc_losses, inc_steps))
    if auto_inc.observe(i, loss, step)
]
auto_inc.end_frame()
assert active_fires == [10], active_fires

unsafe_inc = AutoIncumbentConvergence(
    dict(auto_inc_cfg, auto_spec="fast:4:2:0.1:0.001:0.001"),
    scales=[1.0] * 6,
)
late_better = list(inc_losses)
late_better[14:] = [3.0] * 6
for _ in range(2):
    unsafe_inc.reset_frame()
    for i, (loss, step) in enumerate(zip(late_better, inc_steps)):
        assert not unsafe_inc.observe(i, loss, step)
    unsafe_inc.end_frame()
assert "selected=none" in unsafe_inc.summary()
assert "closest rejected=fast" in unsafe_inc.summary()
unsafe_frontier = unsafe_inc.candidate_frontier()
assert not unsafe_frontier[0]["safe"]
assert "loss_max" in unsafe_frontier[0]["reasons"]
print("  PASS  safe fast rule activates; a late better pose forces full budget")


print("[17] marginal audits need persistence; hard maxima reject immediately")
audit_cfg = dict(
    auto_inc_cfg,
    auto_audit_every=1,
    auto_audit_patience=2,
    auto_spec="only:4:2:0.1:0.001:0.001",
)
audit_steps = [[0.08, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(20)]


def evidence_frame(controller, losses, steps):
    controller.reset_frame()
    fired = [
        i + 1 for i, (loss, step) in enumerate(zip(losses, steps))
        if controller.observe(i, loss, step)
    ]
    controller.end_frame()
    return fired


audited = AutoIncumbentConvergence(audit_cfg, scales=[1.0] * 6)
for _ in range(2):
    assert evidence_frame(audited, inc_losses, audit_steps) == []
assert "selected=only" in audited.summary()
marginal_losses = list(inc_losses)
marginal_losses[14:] = [3.96] * 6
assert evidence_frame(audited, marginal_losses, audit_steps) == []
assert "selected=only" in audited.summary()
assert "audit_deferrals=1" in audited.summary()
assert "audit_streak=1" in audited.summary()
assert evidence_frame(audited, marginal_losses, audit_steps) == []
assert "selected=none" in audited.summary()

hard = AutoIncumbentConvergence(audit_cfg, scales=[1.0] * 6)
for _ in range(2):
    assert evidence_frame(hard, inc_losses, audit_steps) == []
hard_steps = [[0.20, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(20)]
assert evidence_frame(hard, marginal_losses, hard_steps) == []
assert "selected=none" in hard.summary()
assert "audit_deferrals=0" in hard.summary()
print("  PASS  one p90 wobble is retained; persistent or max-bound risk is not")


print("[18] rollback phase discards the discontinuous incumbent pose chain")
rollback = IncumbentConvergence(
    dict(inc_cfg, proposal_after_phase=1), scales=[1.0] * 6
)
rollback.reset_frame()
rollback.observe(0, 10.0, [3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
assert not np.allclose(rollback._current_pose, np.eye(4))
rollback.start_phase(2, "adaptive-diagonal")
rollback.restart_phase_evidence(2)
assert np.allclose(rollback._current_pose, np.eye(4))
assert rollback._first_loss is None and rollback._best_loss is None
assert rollback._last_significant == 2 and rollback._phase_index == 1
rollback.observe(2, 8.0, [0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
assert rollback._first_loss == 8.0
rollback.end_frame()
print("  PASS  diagonal P8 starts from the restored pose, not the rejected trial")


print("[19] automatic incumbent-energy decay calibrates once and freezes")
auto_energy_cfg = {
    "enabled": True,
    "shadow": False,
    "kind": "incumbent_energy",
    "budget": 20,
    "auto_calibration_frames": 2,
    "auto_audit_every": 0,
    "auto_min_proposals": 2,
    "auto_loss_p90": 0.02,
    "auto_loss_max": 0.05,
    "auto_motion_p90": 0.25,
    "auto_motion_max": 0.75,
    "auto_min_saved_fraction": 0.0,
    "auto_proposal_after_phase": 0,
    "patience": 1,
    "check_every": 2,
    "pose_change": 0.1,
    "loss_change": 0.001,
    "progress_min": 0.0,
    "energy_window": 4,
    "energy_patience": 1,
    "energy_phase_relative": False,
    "auto_spec": "strict:0.10,loose:0.25",
}
auto_energy_steps = [
    [
        (1.0 if i < 4 else 0.20 if i < 8 else 0.05)
        * ((-1.0) ** i),
        0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    for i in range(20)
]
auto_energy_losses = [1.0] * 20
auto_energy = AutoIncumbentEnergyConvergence(
    auto_energy_cfg, scales=[1.0] * 6
)
factory_energy = make_windowed_convergence(
    dict(
        auto_energy_cfg,
        auto_enabled=True,
        auto_kind="incumbent_energy",
    ),
    scales=[1.0] * 6,
)
assert isinstance(factory_energy, AutoIncumbentEnergyConvergence)
assert isinstance(factory_energy.RULE_CLASS({}, scales=[1.0] * 6),
                  IncumbentConvergence)
for _ in range(2):
    assert evidence_frame(
        auto_energy, auto_energy_losses, auto_energy_steps
    ) == []
assert "selected=loose" in auto_energy.summary()
assert "decay calibration=frozen" in auto_energy.summary()
assert "periodic audits=disabled" in auto_energy.summary()
assert auto_energy.calibration_seen == 2 and auto_energy.audit_seen == 0
calibration_metrics = auto_energy.metrics(20)
assert calibration_metrics["frames"] == 2
assert calibration_metrics["proposals"] == 2
assert evidence_frame(
    auto_energy, auto_energy_losses, auto_energy_steps
) == [8]
runtime_metrics = auto_energy.metrics(20)
assert runtime_metrics["frames"] == 1
assert runtime_metrics["proposals"] == 1
assert runtime_metrics["proposal_iters"] == [8]
assert len(runtime_metrics["post_motion"]) == 1
assert len(runtime_metrics["post_loss_gain"]) == 1
# No later frame becomes an audit or changes the frozen selection.
assert evidence_frame(
    auto_energy, [1.0] * 16 + [0.5] * 4, auto_energy_steps
) == [8]
assert "selected=loose" in auto_energy.summary()
assert auto_energy.calibration_seen == 2 and auto_energy.audit_seen == 0

# Production logging mode suppresses the candidate frontier and intermediate
# decisions, but still emits one unambiguous frozen choice.
quiet_auto_energy = AutoIncumbentEnergyConvergence(
    dict(
        auto_energy_cfg,
        auto_report_candidates=True,
        auto_final_choice_only=True,
    ),
    scales=[1.0] * 6,
)
quiet_log = StringIO()
with redirect_stdout(quiet_log):
    for _ in range(2):
        assert evidence_frame(
            quiet_auto_energy, auto_energy_losses, auto_energy_steps
        ) == []
quiet_lines = [
    line for line in quiet_log.getvalue().splitlines()
    if "AutoIncumbentEnergyConvergence" in line
]
assert quiet_lines == [
    "[AutoIncumbentEnergyConvergence] final choice: loose (decay=0.25)"
], quiet_lines

# Automatic mode must delegate health handling to the selected incumbent rule.
# Two otherwise-valid proposals remain vetoed by the frozen anomaly; the third
# releases it, while a live drift flag remains absolute after release.
auto_release = make_windowed_convergence(
    dict(
        auto_energy_cfg,
        auto_enabled=True,
        auto_kind="incumbent_energy",
        anomaly_release_after=3,
    ),
    scales=[1.0] * 6,
)
for _ in range(2):
    assert evidence_frame(
        auto_release, auto_energy_losses, auto_energy_steps
    ) == []
auto_release.reset_frame()
released_at = []
for i, (loss, step) in enumerate(zip(auto_energy_losses, auto_energy_steps)):
    if auto_release.observe(i, loss, step):
        if auto_release.health_veto(True, False):
            auto_release.note_barred_proposal()
        else:
            released_at.append(i + 1)
            break
assert len(released_at) == 1, released_at
assert auto_release.barred_proposals == 2
assert auto_release._active_rule.barred_proposals == 2
assert auto_release._active_rule.anomaly_releases == 1
assert auto_release.health_veto(True, True)
auto_release.end_frame()
assert "anomaly_release_after=3, released=1" in auto_release.summary()

unsafe_auto_energy = AutoIncumbentEnergyConvergence(
    auto_energy_cfg, scales=[1.0] * 6
)
unsafe_energy_losses = [1.0] * 16 + [0.5] * 4
for _ in range(2):
    assert evidence_frame(
        unsafe_auto_energy, unsafe_energy_losses, auto_energy_steps
    ) == []
assert "selected=strict" in unsafe_auto_energy.summary()
assert "decay calibration=frozen" in unsafe_auto_energy.summary()
# A no-safe-rule calibration freezes the conservative 0.10 decay rather than
# silently continuing calibration or reverting to full-budget tracking.
assert evidence_frame(
    unsafe_auto_energy, auto_energy_losses, auto_energy_steps
) == [12]
assert unsafe_auto_energy.calibration_seen == 2

# One configured late frame is full-budget evidence, may tighten the frozen
# decay, and can never schedule a second audit.
late_auto_energy = AutoIncumbentEnergyConvergence(
    dict(auto_energy_cfg, auto_late_audit_frame=4), scales=[1.0] * 6
)
for _ in range(2):
    assert evidence_frame(
        late_auto_energy, auto_energy_losses, auto_energy_steps
    ) == []
assert "selected=loose" in late_auto_energy.summary()
assert evidence_frame(
    late_auto_energy, auto_energy_losses, auto_energy_steps
) == [8]
assert evidence_frame(
    late_auto_energy, unsafe_energy_losses, auto_energy_steps
) == []
assert late_auto_energy.audit_seen == 1
assert late_auto_energy.late_audit_done
assert "selected=strict" in late_auto_energy.summary()
assert evidence_frame(
    late_auto_energy, auto_energy_losses, auto_energy_steps
) == [12]
assert evidence_frame(
    late_auto_energy, auto_energy_losses, auto_energy_steps
) == [12]
assert late_auto_energy.audit_seen == 1

# A multi-frame late audit keeps the frozen rule while it gathers every
# requested full tail, then makes exactly one decision at the end of the
# window. One adverse frame must not change the rule immediately.
late3_auto_energy = AutoIncumbentEnergyConvergence(
    dict(
        auto_energy_cfg,
        auto_late_audit_frame=4,
        auto_late_audit_frames=3,
    ),
    scales=[1.0] * 6,
)
for _ in range(2):
    assert evidence_frame(
        late3_auto_energy, auto_energy_losses, auto_energy_steps
    ) == []
assert "selected=loose" in late3_auto_energy.summary()
assert evidence_frame(
    late3_auto_energy, auto_energy_losses, auto_energy_steps
) == [8]
for expected_audits in (1, 2):
    assert evidence_frame(
        late3_auto_energy, unsafe_energy_losses, auto_energy_steps
    ) == []
    assert late3_auto_energy.audit_seen == expected_audits
    assert not late3_auto_energy.late_audit_done
    assert "selected=loose" in late3_auto_energy.summary()
assert evidence_frame(
    late3_auto_energy, unsafe_energy_losses, auto_energy_steps
) == []
assert late3_auto_energy.audit_seen == 3
assert late3_auto_energy.late_audit_done
assert "selected=strict" in late3_auto_energy.summary()
assert "frames 4-6 (done)" in late3_auto_energy.summary()
assert evidence_frame(
    late3_auto_energy, auto_energy_losses, auto_energy_steps
) == [12]
assert late3_auto_energy.audit_seen == 3

try:
    AutoIncumbentEnergyConvergence(
        dict(auto_energy_cfg, auto_audit_every=1), scales=[1.0] * 6
    )
except ValueError as exc:
    assert "calibration-only" in str(exc)
else:
    raise AssertionError("incumbent-energy decay audits were accepted")
try:
    AutoIncumbentEnergyConvergence(
        dict(auto_energy_cfg, auto_late_audit_frame=2), scales=[1.0] * 6
    )
except ValueError as exc:
    assert "after the skipped prefix" in str(exc)
else:
    raise AssertionError("late audit inside calibration was accepted")
try:
    AutoIncumbentEnergyConvergence(
        dict(
            auto_energy_cfg,
            auto_calibration_skip_frames=2,
            auto_late_audit_frame=4,
        ),
        scales=[1.0] * 6,
    )
except ValueError as exc:
    assert "skipped prefix" in str(exc)
else:
    raise AssertionError("late audit inside skip+calibration was accepted")
try:
    AutoIncumbentEnergyConvergence(
        dict(auto_energy_cfg, auto_late_audit_frames=3),
        scales=[1.0] * 6,
    )
except ValueError as exc:
    assert "requires auto_late_audit_frame" in str(exc)
else:
    raise AssertionError("late audit length without a start frame was accepted")
print("  PASS  safe decay freezes; unsafe calibration freezes decay 0.10")


print("[stall] arrived-converged admission, and what it must refuse")
# progress_min rejects frames whose TOTAL improvement is tiny. On Replica the
# pose is already at the optimum when the frame opens, so those are the safest
# frames to stop and the gate blocks exactly them. stall_patience admits them
# on a long run of pose stability - and nothing else.
_stall_base = dict(enabled=True, shadow=True, patience=2, check_every=4,
                   pose_change=0.1, loss_change=0.001, progress_min=0.001,
                   proposal_after_phase=0)
_tiny = [1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]

# Default OFF: a flat frame never proposes, exactly as before this existed.
_off = IncumbentConvergence({**_stall_base, "stall_patience": 0}, scales=[1.0] * 6)
_off.reset_frame()
_fired = [i + 1 for i in range(60) if _off.observe(i, 100.0, _tiny)]
_off.end_frame()
assert _fired == [], _fired
assert _off.progress_rejects > 0 and _off.stall_admits == 0
print("  PASS  stall_patience=0 leaves the flat frame rejected (bit-identical default)")

# ON: the same frame proposes once the stability run reaches the threshold.
_on = IncumbentConvergence({**_stall_base, "stall_patience": 8}, scales=[1.0] * 6)
_on.reset_frame()
_fired = [i + 1 for i in range(60) if _on.observe(i, 100.0, _tiny)]
_on.end_frame()
# 8 = the first check point at which the pose has been still for
# stall_patience iterations, counted from frame start.
assert _fired == [8], _fired
assert _on.stall_admits == 1, _on.stall_admits
print("  PASS  a pose-stable flat frame is admitted at the stability threshold")

# A MOVING pose must never be admitted - that is the stuck/diverging case the
# progress gate exists for. Significant motion every 4th iteration caps the
# stability run at 2, so the check is reached but the admission is refused.
_mov = IncumbentConvergence({**_stall_base, "stall_patience": 8}, scales=[1.0] * 6)
_mov.reset_frame()
_fired = [i + 1 for i in range(60)
          if _mov.observe(i, 100.0,
                          [0.5, 0.0, 0.0, 0.0, 0.0, 0.0] if i % 4 == 1 else _tiny)]
_mov.end_frame()
assert _fired == [], _fired
assert _mov.stall_admits == 0
assert _mov.progress_rejects > 0, "the frame must reach the gate and be refused there"
print("  PASS  a frame whose pose keeps moving is refused, however long it runs")

# A healthy frame takes the normal path; the escape hatch stays unreachable.
# This is why TUM (progress 37-52%) cannot be disturbed by the setting.
_ok = IncumbentConvergence({**_stall_base, "stall_patience": 8}, scales=[1.0] * 6)
_ok.reset_frame()
_losses = [100.0 * (0.5 ** min(i, 5)) for i in range(60)]
_fired = [i + 1 for i in range(60) if _ok.observe(i, _losses[i], _tiny)]
_ok.end_frame()
# 8: the loss plateaus at i=5, freezing _last_significant at 6, and patience=2
# clears at the next check point. What matters is that it went through the
# NORMAL gate - progress was 96.9%, far above the floor.
assert _fired == [8], _fired
assert _ok.stall_admits == 0, "healthy progress must not route through the hatch"
print("  PASS  a frame with real progress uses the normal path, stall_admits=0")

# A threshold below `patience` can never admit anything, so it is a config
# error rather than a no-op.
try:
    IncumbentConvergence({**_stall_base, "stall_patience": 1}, scales=[1.0] * 6)
except ValueError as _e:
    assert "stall_patience" in str(_e), _e
else:
    raise AssertionError("stall_patience below patience must raise")
print("  PASS  stall_patience below patience is rejected at construction")


print("[signals] extra GPU scalars share the existing drain and ring order")
extra_signals = ESSignalBuffer(
    {"enabled": True, "batch": 4}, device="cpu", pose_dim=6, extra_cols=4
)
erot = torch.zeros(3)
etran = torch.zeros(3)
extra_signals.reset_for_frame()
for i in range(4):
    extra_signals.record_pre_step(erot, etran)
    extra_signals.record_post_step(torch.tensor(float(i)), erot, etran)
    if i in (1, 3):
        extra_signals.write_current_extra(
            torch.tensor([0.90 + 0.01 * i, float("nan"),
                          float("nan"), 1.0]))
    extra_signals.note_iteration()
extra_rows = extra_signals.drain_if_full()
assert len(extra_rows) == 4
assert np.isnan(extra_rows[0]["extra"]).all()
assert abs(extra_rows[1]["extra"][0] - 0.91) < 1e-6
assert extra_rows[3]["extra"][3] == 1.0

# A partial phase drain resets both the device write index and its host mirror.
extra_signals.record_pre_step(erot, etran)
extra_signals.record_post_step(torch.tensor(4.0), erot, etran)
extra_signals.write_current_extra(torch.tensor([0.94, 1.0, 1.0, 1.0]))
extra_signals.note_iteration()
phase_row = extra_signals.drain_phase()[0]
assert phase_row["iter"] == 4 and abs(phase_row["extra"][0] - 0.94) < 1e-6
extra_signals.record_pre_step(erot, etran)
extra_signals.record_post_step(torch.tensor(5.0), erot, etran)
extra_signals.write_current_extra(torch.tensor([0.95, 1.0, 1.0, 1.0]))
extra_signals.note_iteration()
post_phase_row = extra_signals.drain_remainder()[0]
assert post_phase_row["iter"] == 5
assert abs(post_phase_row["extra"][0] - 0.95) < 1e-6
print("  PASS  current-row extras survive full and partial drains in order")

print("all checks passed")
