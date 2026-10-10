"""Small, reproducible current preset for full TUM fr1/desk runs.

The implementation remains in ``splatam_precond.py``.  This file only pins
the current settings validated in the TUM run, so running it does not require
the large profiling suite or a long list of shell flags.

``RUN_REP`` is intentionally left to the launcher so every repetition writes
to a different SplaTAM result directory.
"""

import math
import os


# Internal benchmark selector used by
# ``benchmarks/run_splatam_fr1_speedup_breakdown.sh``.  It is deliberately
# not a collection of public on/off knobs: the benchmark exposes one
# cumulative ``--keep`` stage, and this preset remains unchanged when the
# selector is absent.
_breakdown_stage = os.environ.get(
    "_SLAMIO_SPLATAM_BREAKDOWN_STAGE", ""
).strip().lower()
_breakdown_stages = ("baseline", "tracking", "mapping", "reuse", "graphs")
if _breakdown_stage and _breakdown_stage not in _breakdown_stages:
    raise ValueError(
        "_SLAMIO_SPLATAM_BREAKDOWN_STAGE must be one of "
        + ", ".join(_breakdown_stages)
        + f"; got {_breakdown_stage!r}"
    )


# Explicit experiment-only overrides.  The final preset intentionally ignores
# ambient PRE_LR so an old exported tuning variable cannot silently change a
# benchmark.  These names are deliberately separate: callers must opt in to a
# trial, while the no-override path remains bit-for-bit the saved final setup.
_full_lr_override = os.environ.get("SPLATAM_FINAL_PRE_LR", "").strip()
_diag_lr_override = os.environ.get("SPLATAM_FINAL_DIAG_LR", "").strip()
_diag_after_override = os.environ.get(
    "SPLATAM_FINAL_DIAG_AFTER", ""
).strip()
_optimizer_override = os.environ.get(
    "SPLATAM_FINAL_OPTIMIZER", ""
).strip().lower()
_frames_override = os.environ.get("SPLATAM_FINAL_FRAMES", "").strip()
_decay_override = os.environ.get(
    "SPLATAM_FINAL_WCONV_DECAY", ""
).strip()
_diag_sweep_profile = os.environ.get(
    "_SLAMIO_SPLATAM_DIAG_SWEEP", "0"
) not in ("0", "", "false", "False")


_FINAL_ENV = {
    # Full-scene tracking and the accepted pose optimizer.
    "PRECOND": "1",
    "ITERS": "200",
    "FRAMES": "-1",
    # Production CSV choice: full-matrix phase at 0.0015, followed by the
    # model's native Adam rates (translation=rotation=0.002) in the diagonal
    # tail.  Keeping these separate is deliberate: PRE_LR must not leak into
    # the diagonal phase.
    "PRE_LR": "0.0015",
    "PRE_DIAG_LR": "1",
    "PRE_BETA2": "0.98",
    "PRE_HANDOFF": "0",
    "PRE_MAX_STEP": "1.0",
    "PRE_RESTART": "20",
    "PRE_RESTART_M": "off",
    "PRE_DIAG_AFTER": "20",
    "PRE_DEAD_FREEZE": "1",
    "PRE_RAMP": "0",
    "PRE_EXCURSION_TRACE": "",
    "CARRY": "1",
    "AUTO_LR": "0",
    # Current automatic incumbent-energy stopping rule.
    "ES": "0",
    "STOP_REL": "0",
    "FINAL_DENSE": "1",
    "COMMIT_AT_LOSS": "1",
    "WCONV": "1",
    "WCONV_SHADOW": "0",
    "WCONV_KIND": "incumbent_energy",
    "WCONV_BATCH": "4",
    "WCONV_EVERY": "4",
    "WCONV_PATIENCE": "4",
    "WCONV_WINDOW": "4",
    "WCONV_ENERGY_WINDOW": "4",
    "WCONV_ENERGY_PATIENCE": "2",
    # Match the recent A100 TUM run: keep the energy anchor at frame start
    # and release a frozen frame-start anomaly after three valid proposals.
    "WCONV_ENERGY_PHASE_RELATIVE": "0",
    "WCONV_AFTER_PHASE": "1",
    "WCONV_ANOMALY_RELEASE_AFTER": "3",
    "WCONV_POSE_CHANGE": "0.10",
    "WCONV_LOSS_CHANGE": "0.001",
    "WCONV_PROGRESS": "0.001",
    "WCONV_AUTO": "1",
    "WCONV_AUTO_KIND": "incumbent_energy",
    "WCONV_AUTO_SKIP": "8",
    "WCONV_AUTO_CALIB": "12",
    "WCONV_AUTO_AUDIT": "0",
    # Calibrate once on the initial evidence window, then freeze the selected
    # decay for the rest of the sequence.  There is no later full-budget audit.
    "WCONV_AUTO_LATE_AUDIT": "0",
    "WCONV_AUTO_LATE_AUDIT_FRAMES": "1",
    "WCONV_AUTO_MIN_PROPOSALS": "4",
    "WCONV_AUTO_MIN_SAVED": "0",
    "WCONV_AUTO_AFTER_PHASE": "1",
    "WCONV_AUTO_LOSS_P90": "0.03",
    "WCONV_AUTO_LOSS_MAX": "0.07",
    "WCONV_AUTO_MOTION_P90": "1.5",
    "WCONV_AUTO_MOTION_MAX": "2.2",
    # The final preset never selects a decay below 0.30.  This is the complete
    # candidate bank used by its one-shot automatic calibration.
    "WCONV_AUTO_SPEC": "d50:0.50,d40:0.40,d30:0.30",
    # The controller still prints the selected candidate; suppress only the
    # full per-candidate frontier table.
    "WCONV_AUTO_REPORT": "0",
    "WCONV_SWEEP": "",
    # Execution optimizations used by the recorded final rows.
    # Internal runtime switch consumed by splatam.py.  The speedup breakdown
    # overrides it below; ordinary production runs retain the accepted
    # tracking-only backward path.
    "_SLAMIO_SPLATAM_TRACKING_ONLY_BACKWARD": "1",
    "ITERGRAPH": "1",
    "GRAPH": "0",
    "GRAD_REUSE": "2",
    "GRAD_REUSE_CALIBRATE_FRAMES": "0",
    "GRAD_REUSE_HOLD_FRAMES": "20",
    "GRAD_REUSE_WARMUP": "6",
    "GRAD_REUSE_COOLDOWN": "0",
    "GRAD_REUSE_FREEZE_M": "1",
    "GRAD_REUSE_ADAPTIVE": "1",
    "GRAD_REUSE_LOCAL_TRUST": "1",
    "GRAD_REUSE_TRUST_COS": "0.95",
    "GRAD_REUSE_CHECK_EVERY": "1",
    "GRAD_REUSE_STOP_CLOCK": "fresh",
    "GRAD_REUSE_BATCHED_CHECK": "1",
    "GRAD_REUSE_TRUST_LEASE": "2",
    # Remove the rasterizer's count readback in the accepted reuse/graph
    # package.  The speedup waterfall explicitly restores the original
    # synchronous path before the reuse rung so this prerequisite cannot leak
    # into the tracking or mapping measurements.
    "BINCAP": "1",
    # Current self-calibrated mapping budget (25--30 iterations).
    "ADMAP": "1",
    "AM_CALIB_KF": "24",
    "AM_CALIB_SKIP": "8",
    "AM_CALIB_NP": "24",
    "AM_CALIB_NP_SKIP": "8",
    "AM_MIN_ITERS": "25",
    "AM_MAX_ITERS": "30",
}

# Cumulative speedup waterfall.  Gradient-reuse construction and its trust
# checking remain enabled in every stage.  Before the ``reuse`` stage only
# application of the candidate stale gradient is suppressed; this holds the
# checker's real cost constant so the reuse step measures saved renders rather
# than comparing against a control that omitted its validation overhead.
if _breakdown_stage:
    _breakdown_level = _breakdown_stages.index(_breakdown_stage)
    # Hold the stopping policy fixed across the entire timing waterfall.  The
    # production preset still auto-calibrates when no breakdown selector is
    # present, but these arms all use the same accepted decay so calibration
    # cannot change their iteration counts independently.
    _FINAL_ENV.update(
        WCONV_AUTO="0",
        WCONV_DECAY="0.30",
        _SLAMIO_SPLATAM_TRACKING_ONLY_BACKWARD="0",
    )
    if _breakdown_level < _breakdown_stages.index("tracking"):
        _FINAL_ENV.update(PRECOND="0", WCONV="0", WCONV_AUTO="0")
    if _breakdown_level < _breakdown_stages.index("mapping"):
        _FINAL_ENV["ADMAP"] = "0"
    if _breakdown_level < _breakdown_stages.index("reuse"):
        # Keep the real reuse trust checker running, as required by the
        # control, but leave its readback synchronous until reuse is actually
        # applied.  At the reuse rung both this batched readback and the
        # graph-prerequisite binning-capacity path switch on; the later graphs
        # rung therefore measures graph capture alone.
        _FINAL_ENV.update(GRAD_REUSE_BATCHED_CHECK="0", BINCAP="0")
    if _breakdown_level < _breakdown_stages.index("graphs"):
        _FINAL_ENV.update(ITERGRAPH="0", GRAPH="0")

# Controlled diagonal sweep: retain final tracking convergence, graph,
# tracking-only backward, binning-capacity and learning-rate settings, while
# removing the two optimizations the experiment explicitly excludes.
if _diag_sweep_profile:
    _FINAL_ENV.update(
        GRAD_REUSE="0",
        GRAD_REUSE_CALIBRATE_FRAMES="0",
        GRAD_REUSE_HOLD_FRAMES="0",
        ADMAP="0",
        WCONV_AUTO="0",
        WCONV_DECAY="0.10",
    )

# Deliberately assign rather than setdefault: a benchmark preset should not
# silently change because the calling shell happens to export an old tuning
# variable. RUN_REP is not in this table and remains caller-controlled.
os.environ.update(_FINAL_ENV)

# Explicit diagonal-sweep hooks.  These names are intentionally scoped to the
# saved final preset so an ambient PRE_DIAG_AFTER/PRECOND/FRAMES left behind by
# another experiment cannot silently mutate production runs.  A fixed
# diagonal transition and the first-moment restart are moved together, which
# preserves the final preset's D20 semantics at every swept value.
if _diag_after_override:
    try:
        _diag_after = int(_diag_after_override)
        if not 0 < _diag_after < int(_FINAL_ENV["ITERS"]):
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            "SPLATAM_FINAL_DIAG_AFTER must be an integer between 1 and "
            f"{int(_FINAL_ENV['ITERS']) - 1}; got "
            f"{_diag_after_override!r}") from exc
    os.environ["PRE_DIAG_AFTER"] = str(_diag_after)
    os.environ["PRE_RESTART"] = str(_diag_after)

if _optimizer_override:
    if _optimizer_override not in ("slamio", "adam"):
        raise ValueError(
            "SPLATAM_FINAL_OPTIMIZER must be 'slamio' or 'adam'; got "
            f"{_optimizer_override!r}")
    os.environ["PRECOND"] = "1" if _optimizer_override == "slamio" else "0"
    if _optimizer_override == "adam":
        # Adam has no full->diagonal phase boundary.  The final controller's
        # phase>=1 gate is correct for SLAMIO but would leave an Adam control
        # permanently in phase 0, enabled in configuration yet unable to make
        # a proposal.  Start the identical controller in Adam's only phase.
        os.environ["WCONV_AFTER_PHASE"] = "0"
        os.environ["WCONV_AUTO_AFTER_PHASE"] = "0"

if _decay_override:
    try:
        _decay = float(_decay_override)
        if not math.isfinite(_decay) or _decay <= 0.0:
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            "SPLATAM_FINAL_WCONV_DECAY must be a finite positive number; got "
            f"{_decay_override!r}") from exc
    os.environ["WCONV_DECAY"] = _decay_override

if _frames_override:
    try:
        _frames = int(_frames_override)
        if _frames == 0 or _frames < -1:
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            "SPLATAM_FINAL_FRAMES must be -1 or a positive integer; got "
            f"{_frames_override!r}") from exc
    os.environ["FRAMES"] = str(_frames)

if _full_lr_override:
    try:
        if float(_full_lr_override) <= 0:
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            "SPLATAM_FINAL_PRE_LR must be a positive number; got "
            f"{_full_lr_override!r}") from exc
    os.environ["PRE_LR"] = _full_lr_override

# PRE_DIAG_LR already validates either "1"/"auto" (native Adam rates) or an
# explicit trans,rot pair inside splatam_precond.py.  Forward it only when the
# caller deliberately supplied the final-preset override.
if _diag_lr_override:
    os.environ["PRE_DIAG_LR"] = _diag_lr_override

from configs.tum.splatam_precond import config  # noqa: E402

if _breakdown_stage:
    config["tracking"]["_apply_grad_reuse"] = (
        _breakdown_stages.index(_breakdown_stage)
        >= _breakdown_stages.index("reuse")
    )
    config["run_name"] += f"_speedbreak_{_breakdown_stage}"
