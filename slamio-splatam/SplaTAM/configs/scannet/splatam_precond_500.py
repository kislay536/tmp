# FULL-MATRIX POSE PRECONDITIONER on ScanNet. Both arms, one file.
#
#   SCENE_NUM=1 PRECOND=1 python scripts/splatam.py configs/scannet/splatam_precond_500.py
#   SCENE_NUM=1 PRECOND=0 python scripts/splatam.py configs/scannet/splatam_precond_500.py
#
# ONE FILE FOR BOTH ARMS, for the same reason as configs/tum/splatam_precond.py:
# the only thing that may differ between the arms is the update rule. PRECOND=0
# is the matched Adam control, not the main-branch baseline.
#
# DO NOT USE THE MAIN-BRANCH 500-FRAME RUN AS THIS ARM'S CONTROL. This branch
# differs from main by ~2350 lines of CUDA rasterizer, so a main-vs-branch
# comparison measures the rasterizer as much as the preconditioner. That is the
# uncontrolled-difference failure the A/B counterbalancing note already caught
# once. The main run is a separate reference, not a control.
#
# BASE IS STOCK SCANNET, NOT THE TUM PRECOND LADDER. configs/tum/splatam_precond.py
# inherits a whole TUM tuning stack - pixel sampling, early stop, binning
# capacity, the CUDA graphs - through splatam_adam_dose. None of that is
# calibrated for ScanNet, and every one of those would be a second variable on
# top of the update rule. So this config takes configs/scannet/splatam_500.py
# (stock SplaTAM: 100 tracking iters, 30 mapping iters, stock lrs) and adds
# nothing but the preconditioner block. The preconditioner parameters
# themselves are copied verbatim from the TUM config, same env-var interface
# and same defaults.
#
# CUDA GRAPH: the ITERATION graph (ITERGRAPH=1, ported below) is available
# and, per TUM's own accepted config, compatible with both the preconditioner
# and gradient reuse. It is off by default, same as TUM's own bare default -
# nothing here turns it on unasked. The WHOLE-FRAME cuda_graph is a different,
# separate mechanism this file has never wired in and still does not; it is
# also the one genuinely incompatible with reuse (scripts/splatam.py raises
# if both are enabled together), which is not a concern as long as ITERGRAPH
# is what gets used, not that other graph.
#
# EXPECT A NULL AT 100 ITERATIONS, AND DO NOT READ IT AS A REFUTATION. The
# method's claim is that it holds accuracy at a budget where Adam does not; the
# MonoGS budget finding puts the effect at 5-7x in Adam's favour-free zone of
# cap 20-30 and NULL at full budget, because a saturated arm has nothing left
# to win. ScanNet stock is 100 tracking iterations, i.e. saturated. This run is
# a SMOKE TEST: does it run on a new dataset, does it diverge, and what does
# the 6x6 solve cost per iteration against a ~23 ms ScanNet render. The
# informative comparison is the same file at ITERS=20-40.
#
# AND REMEMBER THE BIMODALITY. SplaTAM tracking at reduced budget is bimodal -
# identical config and seed gave 17.09 cm and 3.70 cm on TUM. A single run of
# either arm is one draw from a coin flip. Once this smoke test passes, the
# comparison needs n>=5 per arm and a SUCCESS RATE, not a mean ATE.
from configs.scannet.splatam_500 import config as _base
import copy
import os

_on = os.environ.get("PRECOND", "1") not in ("0", "", "false", "False")

config = copy.deepcopy(_base)

# ITERS overrides the tracking budget, which is the axis that actually matters
# here. Left at the stock 100 by default so the first run is the like-for-like
# one that was asked for.
_iters = int(os.environ.get("ITERS", str(config["tracking"]["num_iters"])))
config["tracking"]["num_iters"] = _iters

# RUN_REP keeps each replicate's params.npz. Without it every rep writes to one
# directory and overwrites the last, so only the final run's trajectory
# survives - which is how an earlier 5/5 sweep lost trajectories it had already
# paid for.
# FRAMES overrides the sequence length so the same arm can be run at full
# length against the 1807-frame main baseline (9.91 cm). -1 means the whole
# sequence. The default is the 500 the base config carries, so every existing
# run name is unchanged.
_frames = int(os.environ.get("FRAMES", str(_base["data"]["num_frames"])))
config["data"]["num_frames"] = _frames
config["data"]["end"] = _frames if _frames > 0 else -1

_rep = os.environ.get("RUN_REP", "")
# The diagonal transition MUST appear in the run name too. Without it a
# PRE_DIAG_AFTER sweep writes every arm to the same directory and each run
# silently overwrites the last.
_diag = int(os.environ.get("PRE_DIAG_AFTER", "0"))
# The frame count lives in the base name ("..._500"); swap it for whatever
# FRAMES asked for so a full-length run cannot overwrite a 500-frame one.
_stem = _base["run_name"]
if _stem.endswith("_500"):
    _stem = _stem[:-4]
_stem = _stem + "_" + ("full" if _frames < 0 else str(_frames))
# PRE_LR MUST BE IN run_name. run_name keys the output directory and
# params.npz, so an LR ladder without it points every arm at ONE directory and
# each run clobbers the previous arm's parameters. The per-run LOG name is
# distinct (run_adam_dose.sh puts lr in its tag) so the printed ATE survives,
# which is exactly what makes this silent. Same bug class as RUN_REP (f1d31ea),
# PRE_SHRINK (403b060), PRE_ADAPTIVE_DIAG and PRE_DIAG_LR.
_lrtag = os.environ.get("PRE_LR", "")
config["run_name"] = (f"{_stem}_precond{'on' if _on else 'off'}"
                      f"_i{_iters}"
                      + (f"_lr{_lrtag}" if (_on and _lrtag) else "")
                      + (f"_d{_diag}" if _diag else "")
                      + (f"_r{_rep}" if _rep else ""))

# Preconditioner parameters copied verbatim from configs/tum/splatam_precond.py
# so "the same parameters" is literal. Defaults: Stage 1 (tangent=1, the SE(3)
# tangent parameterisation), transport OFF, carry ON, beta2 0.95, lr 0.002,
# tau 0.05, refactor_every 10. Stopping is disabled by default (stop_rel_grad
# 0.0) so the budget is exactly _iters in both arms.
config["tracking"]["preconditioner"] = dict(
    enabled=_on,
    auto_lr=os.environ.get("AUTO_LR", "0") not in ("0", "", "false", "False"),
    stop_rel_grad=float(os.environ.get("STOP_REL", "0.0")),
    stop_mode=os.environ.get("STOP_MODE", "rel"),
    stop_patience=int(os.environ.get("STOP_PATIENCE", "10")),
    final_dense=os.environ.get("FINAL_DENSE", "0") not in ("0", "", "false", "False"),
    stop_min_iters=int(os.environ.get("STOP_MIN", "10")),
    stop_ref=os.environ.get("STOP_REF", "frame"),
    stop_check_every=int(os.environ.get("STOP_EVERY", "5")),
    tangent=os.environ.get("PRECOND_TANGENT", "1") not in ("0", "", "false"),
    lr=float(os.environ.get("PRE_LR", "0.002")),
    beta1=float(os.environ.get("PRE_BETA1", "0.9")),
    beta2=float(os.environ.get("PRE_BETA2", "0.95")),
    tau=float(os.environ.get("PRE_TAU", "0.05")),
    shrink=float(os.environ.get("PRE_SHRINK", "1.0")),
    sample_metric=os.environ.get("PRE_SAMPLES", "0") not in ("0", "", "false", "False"),
    stop_improve=float(os.environ.get("STOP_IMPROVE", "0.01")),
    handoff=int(os.environ.get("PRE_HANDOFF", "0")),
    m0_iso=os.environ.get("PRE_M0", "") == "iso",
    bfgs=os.environ.get("BFGS", "0") not in ("0", "", "false", "False"),
    bfgs_max_cond=float(os.environ.get("BFGS_COND", "20.0")),
    bfgs_curv_eps=float(os.environ.get("BFGS_CURV", "1e-2")),
    bfgs_max_growth=float(os.environ.get("BFGS_GROWTH", "4.0")),
    bfgs_b0_from_lrs=os.environ.get("BFGS_B0", "lrs") == "lrs",
    refactor_every=10,
    max_step_mult=float(os.environ.get("PRE_MAX_STEP", "10.0")),
    restart_at=int(os.environ.get("PRE_RESTART", "0")),
    restart_m=os.environ.get("PRE_RESTART_M", "off"),
    profile_every=int(os.environ.get("PRE_PROFILE_EVERY", "0")),
    # LR_SERIES=1: print the raw per-frame IT/FRAME, REL-GRAD, K series in
    # the summary. Off by default - lr_ladder.sh sets it for the runs it
    # hands to profiling/lr_proxy_replay.py; nobody else needs it.
    log_series=os.environ.get("LR_SERIES", "0") not in ("0", "", "false", "False"),
    excursion_trace={},
    dead_freeze=os.environ.get("PRE_DEAD_FREEZE", "1") not in ("0", "", "false", "False"),
    drift_mult=float(os.environ.get("DRIFT_MULT", "0")),
    drift_warmup=int(os.environ.get("DRIFT_WARMUP", "50")),
    ramp_from=0,
    ramp_to=0,
    diag_after=_diag,
    stop_anchor=os.environ.get("STOP_ANCHOR", "min"),
    carry=os.environ.get("CARRY", "1") not in ("0", "", "false", "False"),
    transport=os.environ.get("TRANSPORT", "0") not in ("0", "", "false", "False"),
    anom_mult=float(os.environ.get("ANOM_MULT", "2.0")),
    ref_decay=float(os.environ.get("REF_DECAY", "0.98")),
    barred_admit=float(os.environ.get("BARRED_ADMIT", "0.0")),
)

# ScanNet automatic incumbent-energy calibration defaults. The broad d40
# frontier remains available, but the final full-sequence run showed that the
# earlier 5%/12% loss limits could accept d40 while degrading downstream
# reconstruction quality. These limits reject that measured d40 evidence
# while still admitting the validated d25 evidence. Environment overrides in
# scripts/splatam.py remain authoritative for explicit ablations.
config["tracking"]["windowed_convergence"] = dict(
    config["tracking"].get("windowed_convergence", {}),
    auto_spec=(
        "d40:0.40,d35:0.35,d30:0.30,d25:0.25,d20:0.20,"
        "d15:0.15,d10:0.10,d075:0.075,d05:0.05,d025:0.025"
    ),
    auto_loss_p90=0.025,
    auto_loss_max=0.05,
    auto_motion_p90=1.5,
    auto_motion_max=2.2,
)

# ITERATION GRAPH (ITERGRAPH=1) + BINNING CAPACITY + MASK_MULTIPLY_LOSS,
# ported from configs/tum/splatam_precond.py's chain verbatim - same env
# vars, same defaults (margin=2.0, min_capacity=0, warmup_iters=3). None of
# the three is scene-calibrated for ScanNet beyond inheriting TUM's values;
# all default to TUM's own bare-default behaviour (binning/mask_multiply
# unconditionally on - they're not graph-only, TUM's accepted config always
# runs with them; iteration_graph itself off unless ITERGRAPH=1), so this is
# additive, not a behaviour change for the existing 500-frame smoke test's
# own defaults.
#
# mask_multiply_loss IS A HARD PREREQUISITE TOO, same class of requirement
# as binning capacity, discovered by a real capture crash: get_loss's normal
# x[mask].sum() is boolean-mask indexing, which CUDA graph capture rejects
# outright ("operation not permitted when stream is capturing"). TUM's chain
# replaces it with torch.where(mask, x, 0).sum() - not bit-identical, but the
# validated graph-safe equivalent (see configs/tum/splatam_es_graph_eps_
# pixel075_maskmul.py). This is tracking-only, matching TUM's accepted
# config - mapping's own mask_multiply_loss is a separate, unrelated
# experimental arm there (splatam_mapsync_a/b.py), not part of the chain
# splatam_precond.py actually inherits.
config["tracking"]["mask_multiply_loss"] = True

_itergraph = os.environ.get("ITERGRAPH", "0") not in ("0", "", "false", "False")
config["tracking"].setdefault("iteration_graph", {})["enabled"] = _itergraph
config["tracking"]["iteration_graph"].setdefault("warmup_iters", 5)
if _itergraph:
    config["run_name"] += "_ig"

config["tracking"]["binning_capacity"] = dict(
    enabled=os.environ.get("BINCAP", "1") not in ("0", "", "false", "False"),
    margin=float(os.environ.get("BIN_MARGIN", "2.0")),
    min_capacity=int(os.environ.get("BIN_MIN", "0")),
    warmup_iters=3,
)

# Adaptive mapping is opt-in for ScanNet so existing preconditioner smoke
# tests retain the stock 30-iteration mapping schedule.  When enabled, use
# the same calibrate-then-freeze interface as TUM and Replica: mapping stays
# at its native ceiling during the calibration window, then the fitted
# depth/colour/new-point references determine a bounded budget.
_admap_env = os.environ.get("ADMAP")
if _admap_env is not None:
    _admap_on = _admap_env not in ("0", "", "false", "False")
    _am_cal = int(os.environ.get("AM_CALIB_KF", "24"))
    _am_skip = int(os.environ.get("AM_CALIB_SKIP", "8"))
    _am_np_cal = int(os.environ.get("AM_CALIB_NP", "24"))
    _am_np_skip = int(os.environ.get("AM_CALIB_NP_SKIP", "8"))
    _am_min = int(os.environ.get("AM_MIN_ITERS", "25"))
    if min(_am_cal, _am_skip, _am_np_cal, _am_np_skip) < 0:
        raise ValueError("adaptive-mapping calibration counts must be >= 0")

    config["adaptive_mapping"] = dict(
        enabled=_admap_on,
        min_iters=_am_min,
        max_iters=int(os.environ.get(
            "AM_MAX_ITERS", str(config["mapping"]["num_iters"]))),
        depth_error_ref=float(os.environ.get("AM_DEPTH_REF", "0.05")),
        color_error_ref=float(os.environ.get("AM_COLOR_REF", "0.05")),
        n_new_pts_ref=int(os.environ.get("AM_NPTS_REF", "300")),
        calibration_keyframes=_am_cal,
        calibration_skip=_am_skip,
        calibrate_new_pts_frames=_am_np_cal,
        calibrate_new_pts_skip=_am_np_skip,
    )
    if _admap_on:
        config["run_name"] += f"_admapcal{_am_cal}_npcal{_am_np_cal}"
        if _am_skip != 8:
            config["run_name"] += f"s{_am_skip}"
        if _am_np_skip != 8:
            config["run_name"] += f"nps{_am_np_skip}"
        if _am_min != 25:
            config["run_name"] += f"_ammin{_am_min}"
    else:
        config["run_name"] += "_noadmap"
