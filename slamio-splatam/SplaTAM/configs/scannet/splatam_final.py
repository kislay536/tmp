"""Minimal full-scene ScanNet scene0059_00 benchmark preset.

The implementation remains in ``splatam_precond_500.py``.  This module only
pins the saved ScanNet policy so repetitions cannot inherit stale shell
variables.  ``RUN_REP`` remains launcher-controlled to preserve every result.
"""

import os


_FINAL_ENV = {
    "SCENE_NUM": "1",
    "PRECOND": "1",
    "ITERS": "100",
    "FRAMES": "-1",
    "PRE_LR": "0.002",
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
    "BARRED_ADMIT": "0.02",
    # Safer ScanNet automatic incumbent-energy calibration.
    "ES": "0",
    "STOP_REL": "0",
    "FINAL_DENSE": "1",
    "COMMIT_AT_LOSS": "1",
    "WCONV": "1",
    "WCONV_SHADOW": "0",
    "WCONV_KIND": "incumbent_energy",
    "WCONV_BATCH": "4",
    "WCONV_EVERY": "4",
    "WCONV_PATIENCE": "2",
    "WCONV_WINDOW": "4",
    "WCONV_ENERGY_WINDOW": "4",
    "WCONV_ENERGY_PATIENCE": "2",
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
    "WCONV_AUTO_LATE_AUDIT": "900",
    "WCONV_AUTO_LATE_AUDIT_FRAMES": "3",
    "WCONV_AUTO_MIN_PROPOSALS": "4",
    "WCONV_AUTO_MIN_SAVED": "0",
    "WCONV_AUTO_AFTER_PHASE": "1",
    "WCONV_AUTO_LOSS_P90": "0.025",
    "WCONV_AUTO_LOSS_MAX": "0.05",
    "WCONV_AUTO_MOTION_P90": "1.5",
    "WCONV_AUTO_MOTION_MAX": "2.2",
    # Do not let machine-specific calibration retreat below the validated
    # ScanNet d25 operating point. If none of these candidates passes the
    # safety filter, the controller's strict fallback within this restricted
    # frontier is d25 rather than d20 or a still smaller decay.
    "WCONV_AUTO_SPEC": "d40:0.40,d35:0.35,d30:0.30,d25:0.25",
    # Keep the selected-candidate line but omit the complete frontier table.
    "WCONV_AUTO_REPORT": "0",
    "WCONV_SWEEP": "",
    # Capture-safe tracking and adaptive gradient reuse.
    "ITERGRAPH": "1",
    "GRAPH": "0",
    "BINCAP": "1",
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
    # Self-calibrated 25--30 mapping-iteration budget.
    "ADMAP": "1",
    "AM_CALIB_KF": "24",
    "AM_CALIB_SKIP": "8",
    "AM_CALIB_NP": "24",
    "AM_CALIB_NP_SKIP": "8",
    "AM_MIN_ITERS": "25",
    "AM_MAX_ITERS": "30",
}

os.environ.update(_FINAL_ENV)

from configs.scannet.splatam_precond_500 import config  # noqa: E402,F401
