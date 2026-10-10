# FULL-MATRIX POSE PRECONDITIONER ON REPLICA. Second scene.
#
#   PRECOND=1 SCENE=room0 python scripts/splatam.py configs/replica/splatam_precond.py
#   PRECOND=0 SCENE=room0 python scripts/splatam.py configs/replica/splatam_precond.py
#
# WHY A SECOND SCENE, AND WHY IT MATTERS MORE THAN ANOTHER TUM SWEEP. Every
# number so far is freiburg1_desk. The stopping threshold, PRE_LR, tau and the
# anomaly bound were all chosen against one sequence's statistics, so "2.6x on
# fr1_desk" and "2.6x on TUM" are different claims and only the second is worth
# writing. This is also where the result is most likely to break.
#
# AND THE TAIL IS THE OPEN QUESTION. On fr1_desk the ATE distribution is not a
# reliable value with a rare glitch - it is mostly 3.2-3.5 cm with occasional
# runs at 6-15 cm, and NO threshold removed them: 10x failed ~1 in 4, and 15x
# failed too once it had more than five samples. Replica says whether that tail
# belongs to the scene or to the method. A clean 5/5 here points at fr1_desk; a
# similar tail points at the preconditioner.
#
# REPLICA IS A DIFFERENT REGIME FROM TUM, in ways that matter here:
#   - synthetic and noise-free, so gradients are cleaner and the metric M is
#     estimated from a better-conditioned signal
#   - tracking_iters 40 and mapping_iters 60 by default, against TUM's 200/30 -
#     mapping DOMINATES, so tracking-side savings are diluted much more
#   - 2000 frames per scene against 592
# Do not carry TUM's tuned values over as if they were universal. PRE_LR in
# particular was fitted to TUM's gradient scale; the |g|/|g0| criterion is
# scale-free across frames but the STEP SIZE is not scale-free across datasets.
from configs.replica.splatam import config as _base
import copy
import os

_on = os.environ.get("PRECOND", "1") not in ("0", "", "false", "False")
_scene = os.environ.get("SCENE", "room0")
_frames = int(os.environ.get("FRAMES", "-1"))
_iters = int(os.environ.get("ITERS", "40"))
_rep = os.environ.get("RUN_REP", "")
_commit_at_loss = int(os.environ.get("COMMIT_AT_LOSS", "0"))
_restart = int(os.environ.get("PRE_RESTART", "0"))
_restart_m = os.environ.get("PRE_RESTART_M", "off")
_diag_after = int(os.environ.get("PRE_DIAG_AFTER", "0"))
if _restart < 0:
    raise ValueError("PRE_RESTART must be >= 0")
if _diag_after < 0:
    raise ValueError("PRE_DIAG_AFTER must be >= 0")

config = copy.deepcopy(_base)
config["use_wandb"] = False
config["data"]["sequence"] = _scene
config["data"]["num_frames"] = _frames
config["tracking"]["num_iters"] = _iters
# MAP_ITERS: fixed mapping iteration budget, independent of the tracking
# budget (_iters) - for the budget-vs-quality ablation (does mapping need
# fewer iterations than the shipped default). Only meaningful with ADMAP=0
# (adaptive_mapping.max_iters below inherits config["mapping"]["num_iters"],
# so this overrides both the fixed-cap path and adaptive's ceiling).
_map_iters = int(os.environ.get("MAP_ITERS", str(config["mapping"]["num_iters"])))
config["mapping"]["num_iters"] = _map_iters
config["run_name"] = (f"{_scene}_precond{'on' if _on else 'off'}"
                      f"_i{_iters}_f{_frames}")
if _map_iters != int(_base["mapping"]["num_iters"]):
    config["run_name"] += f"_mi{_map_iters}"
# Same reason as RUN_REP: the shell tags gamma into the LOG name, but SplaTAM
# keys its output directory and params.npz on run_name, so without this the
# gamma arms overwrite each other and their trajectories are lost.
_shrink = float(os.environ.get("PRE_SHRINK", "1.0"))
if _on and _shrink != 1.0:
    config["run_name"] += f"_s{_shrink:g}"
# SAME TAGGING RULE AS TUM, and for the same reason: run_name keys the output
# directory and params.npz, so an untagged knob silently overwrites the arm it
# is meant to be compared against.
if _on and os.environ.get("STOP_MODE", "rel") == "best":
    config["run_name"] += f"_best{os.environ.get('STOP_PATIENCE', '10')}"
if _on and os.environ.get("FINAL_DENSE", "0") not in ("0", "", "false", "False"):
    config["run_name"] += "fd"
if _on and os.environ.get("STOP_MIN", "10") != "10":
    config["run_name"] += f"m{os.environ.get('STOP_MIN')}"
if _on and os.environ.get("PRE_HANDOFF", "0") != "0":
    config["run_name"] += f"_ho{os.environ.get('PRE_HANDOFF')}"
_msv = os.environ.get("PRE_MAX_STEP", "10.0")
if _on and _msv not in ("10.0", "10", ""):
    config["run_name"] += f"_ms{_msv}"
if _on and _restart:
    config["run_name"] += f"_rst{_restart}"
    if _restart_m != "off":
        config["run_name"] += _restart_m
# PRE_DIAG_LR - the diagonal tail runs on THIS config's own Adam rates instead
# of sharing PRE_LR with the full-matrix phase. The full rationale is in
# configs/tum/splatam_precond.py; what matters here is that Replica is the
# scene the flag was written for. Its two rates are 5x apart -
# cam_trans 0.002 against cam_unnorm_rots 0.0004 - where TUM's are identical at
# 0.002, so a single shared rate is exact on TUM and a BLEND here. That makes
# the shared rate a confound in exactly the comparison this file exists to run.
#
# Order is [trans]*3 + [rot]*3: quat_trans_grad_to_tangent returns
# cat([g_rho, g_theta]) and se3_exp_capturable reads xi[:3], xi[3:].
#
# Parsed after the deepcopy because it reads the base config's own lrs, unlike
# _diag_after which is pure env and is resolved above.
_diag_lr_env = os.environ.get("PRE_DIAG_LR", "0")
_diag_lr = None
_diag_lr_tag = ""
if _diag_lr_env not in ("0", "", "false", "False"):
    if os.environ.get("PRECOND_TANGENT", "1") in ("0", "", "false", "False"):
        # dim=7 there (4 quat + 3 trans) and the coordinates are
        # QUATERNION-FIRST, so six translation-first rates are both the wrong
        # length and the wrong order. The class would reject the length with a
        # message that names neither this flag nor the reason.
        raise ValueError(
            "PRE_DIAG_LR requires the SE(3) tangent (PRECOND_TANGENT=1); the "
            "7-parameter quaternion path takes neither this length nor this "
            "coordinate order.")
    if not _diag_after:
        raise ValueError(
            "PRE_DIAG_LR needs a diagonal tail to apply to: set "
            "PRE_DIAG_AFTER=N. With the full matrix running the whole frame "
            "there is no tail for these rates to reach, so the k port arm is "
            "unaffected by this flag.")
    if _diag_lr_env in ("1", "auto", "true", "True"):
        _dlr_t = float(config["tracking"]["lrs"]["cam_trans"])
        _dlr_r = float(config["tracking"]["lrs"]["cam_unnorm_rots"])
        _diag_lr_tag = "_dlr"
    else:
        try:
            _dlr_t, _dlr_r = (float(x) for x in _diag_lr_env.split(","))
        except ValueError:
            raise ValueError(
                "PRE_DIAG_LR must be 1 (derive from this config's own Adam "
                "lrs) or an explicit trans,rot pair such as 0.002,0.0004 - "
                f"got {_diag_lr_env!r}")
        _diag_lr_tag = f"_dlr{_dlr_t:g}-{_dlr_r:g}"
    if _dlr_t < 0 or _dlr_r < 0:
        raise ValueError("PRE_DIAG_LR rates must be >= 0")
    _diag_lr = [_dlr_t] * 3 + [_dlr_r] * 3

# PRE_BETA2 GOES IN run_name, exactly as it does in the TUM config.
# INSTANCE NINETEEN OF THE UNTAGGED-ARM BUG: this file READ PRE_BETA2 (it is
# passed to the preconditioner below) but never tagged it, while
# configs/tum/splatam_precond.py did. run_name keys the output directory and
# params.npz, so a beta2 sweep on Replica wrote every arm to ONE directory and
# each run silently overwrote the last one's trajectory artifacts. The ATEs
# printed to stdout were unaffected, which is exactly what makes this class of
# bug survive - the numbers look fine and only the artifacts are gone.
if _on and os.environ.get("PRE_BETA2", "0.95") != "0.95":
    config["run_name"] += f"_b2{os.environ.get('PRE_BETA2')}"
if _on and _diag_after:
    config["run_name"] += f"_diag{_diag_after}"
if os.environ.get(
        "WCONV_ENERGY_PHASE_RELATIVE", "1"
        ) in ("0", "", "false", "False"):
    # Keep frame-relative stopping artifacts separate from the default.
    config["run_name"] += "_eprF"
_anomaly_release_after = int(os.environ.get(
    "WCONV_ANOMALY_RELEASE_AFTER", "0"
))
if _on and _anomaly_release_after > 0:
    config["run_name"] += f"_anrel{_anomaly_release_after}"
if _on and _diag_lr_tag:
    # run_name keys the output directory and params.npz - an untagged arm
    # overwrites the one it should be compared against.
    config["run_name"] += _diag_lr_tag
if _on and os.environ.get("PRE_DEAD_FREEZE", "1") in ("0", "", "false", "False"):
    config["run_name"] += "_nodf"
if _commit_at_loss == 1:
    config["run_name"] += "_cal"
elif _commit_at_loss >= 2:
    config["run_name"] += "_cal2"
if _rep:
    config["run_name"] += f"_r{_rep}"

# The ladder opts that the Replica rungs use. Both graphs are available and
# both default OFF: the step graph (GRAPH=1) measured a net loss on TUM
# (+3.6% per iteration, more than cancelled by ~25% more iterations), and the
# iteration graph (ITERGRAPH=1) is newly possible but unvalidated end-to-end.
_t = config["tracking"]
_t["mask_multiply_loss"] = True
_t["clear_gaussian_grads"] = True
_t["pixel_sample"] = dict(enabled=True, sample_ratio=0.75,
                          full_start_ratio=0.0, full_end_ratio=1.0,
                          gradient_frac=0.5)
# ITERGRAPH=1 captures the WHOLE tracking iteration, which is what
# Replica's own ladder rungs use and is worth far more than the step
# graph. It used to be incompatible - it captures optimizer.step(), which
# the preconditioner replaces - but with it on the step graph is disabled
# and the update runs eagerly INSIDE the outer capture, and that update is
# now capture-safe. Default off because the combination is unvalidated
# end-to-end; if it works, it is the biggest remaining speedup here.
_itergraph = os.environ.get("ITERGRAPH", "0") not in ("0", "", "false", "False")
_t["iteration_graph"] = dict(enabled=_itergraph, warmup_iters=5)
if _itergraph:
    config["run_name"] += "_ig"
_graph = os.environ.get("GRAPH", "0") not in ("0", "", "false", "False")
_t["cuda_graph"] = dict(enabled=_graph, warmup_iters=3)
if _graph:
    config["run_name"] += "_g"

# Replica's own binning settings (margin 1.5, min 400k), not TUM's. TUM needed
# 2.0 because the preconditioner moves the pose enough WITHIN a frame to bring
# new Gaussians into view after capacity is fixed from the warmup iterations -
# margin 1.25 overflowed at frame 303. Replica's scenes are smaller, so start
# from its own values and raise if the ovf column stops reading 0.
_t["binning_capacity"] = dict(
    # BINCAP=0 DISABLES IT ENTIRELY, for measuring the GRAPH PACKAGE rather
    # than the graph's marginal value.
    #
    # Binning capacity is a HARD PREREQUISITE for the iteration graph: without
    # a fixed capacity the rasterizer reads num_rendered back to the host to
    # size its binning buffer, and a capture cannot contain a host sync. So it
    # exists BECAUSE of the graph - but it also helps on its own, by removing
    # that readback, and the base arm has been carrying it while getting no
    # graph. That makes the measured graph-on-vs-graph-off delta the graph's
    # MARGINAL value given binning, not the value of the package.
    #
    #   BINCAP=0 GRAPH=off   the true baseline - neither
    #   BINCAP=1 GRAPH=off   the current base arm
    #   BINCAP=1 GRAPH=iter  the current _ig arm
    #
    # package = (iter, on) against (off, off); the middle row decomposes it.
    #
    # Disabling is SAFE, unlike undersizing it: with it off the rasterizer
    # sizes the buffer exactly from num_rendered, so there is no overflow risk
    # at all - only the sync it was added to remove.
    enabled=os.environ.get("BINCAP", "1") not in ("0", "", "false", "False"),

    margin=float(os.environ.get("BIN_MARGIN", "1.5")),
    min_capacity=int(os.environ.get("BIN_MIN", "400000")),
    warmup_iters=3,
)

# EARLY STOPPING OFF when the preconditioner does its own stopping - two
# criteria firing on the same loop is not an experiment. STOP_REL=0 falls back
# to the early-stop machinery.
_t["early_stop"] = dict(enabled=False, min_iters=12, loss_eps=1e-4,
                        patience=3, pose_eps=0.0, retune_every=0,
                        batched=dict(pose_at_loss=_commit_at_loss))

# Replica's safer automatic stopping frontier. Keep calibration automatic,
# but cap its candidate set at d30 rather than allowing the d40 rule used by
# the faster full run. The 2%/5% loss and 0.25/0.75 motion limits are the
# filters that selected d30 on the validated 500-frame calibration run.
# Explicit environment values still override every field in splatam.py.
_t["windowed_convergence"] = dict(
    _t.get("windowed_convergence", {}),
    auto_spec=(
        "d30:0.30,d25:0.25,d20:0.20,d15:0.15,d10:0.10,"
        "d075:0.075,d05:0.05,d025:0.025"
    ),
    auto_loss_p90=0.02,
    auto_loss_max=0.05,
    auto_motion_p90=0.25,
    auto_motion_max=0.75,
)

# ADAPTIVE MAPPING.  The fixed defaults are Replica references measured by
# configs/replica/splatam_room0_itergraph_admap.py, not TUM's.  Setting either
# AM_CALIB_KF or AM_CALIB_NP enables the same calibrate-then-freeze path as the
# TUM preconditioner config: mapping stays at max_iters until both requested
# windows finish, then the fitted references drive the adaptive budget.
#
# It matters more here than on TUM. Mapping is 60 iterations against tracking's
# 40, so mapping DOMINATES in-loop time - a tracking-side speedup alone is
# diluted, and the two together are what the comparison should show.
_admap_enabled = os.environ.get("ADMAP", "1") not in (
    "0", "", "false", "False"
)
_am_cal = int(os.environ.get("AM_CALIB_KF", "0"))
_am_skip = int(os.environ.get("AM_CALIB_SKIP", "8"))
_am_np_cal = int(os.environ.get("AM_CALIB_NP", "0"))
_am_np_skip = int(os.environ.get("AM_CALIB_NP_SKIP", "8"))
_am_min = int(os.environ.get("AM_MIN_ITERS", "25"))
if min(_am_cal, _am_skip, _am_np_cal, _am_np_skip) < 0:
    raise ValueError("adaptive-mapping calibration counts must be >= 0")

config["adaptive_mapping"] = dict(
    enabled=_admap_enabled,
    # AM_MIN_ITERS: the floor under the computed budget, regardless of how
    # low novelty reads. Raising it is the direct "less aggressive" lever -
    # it doesn't touch the novelty calibration (AM_DEPTH_REF/AM_COLOR_REF/
    # AM_NPTS_REF), just how far the budget is allowed to fall.
    min_iters=_am_min,
    max_iters=int(os.environ.get("AM_MAX_ITERS", str(config["mapping"]["num_iters"]))),   # 60 on Replica
    depth_error_ref=float(os.environ.get("AM_DEPTH_REF", "0.003")),
    color_error_ref=float(os.environ.get("AM_COLOR_REF", "0.02")),
    n_new_pts_ref=int(os.environ.get("AM_NPTS_REF", "300")),
    calibration_keyframes=_am_cal,
    calibration_skip=_am_skip,
    calibrate_new_pts_frames=_am_np_cal,
    calibrate_new_pts_skip=_am_np_skip,
)
if _admap_enabled:
    if _am_cal > 0 or _am_np_cal > 0:
        config["run_name"] += f"_admapcal{_am_cal}_npcal{_am_np_cal}"
        if _am_skip != 8:
            config["run_name"] += f"s{_am_skip}"
        if _am_np_skip != 8:
            config["run_name"] += f"nps{_am_np_skip}"
    else:
        config["run_name"] += "_admapfixed"
    if _am_min != 15:
        config["run_name"] += f"_ammin{_am_min}"
else:
    config["run_name"] += "_noadmap"
config["mapping"]["adaptive_pruning"] = dict(enabled=False)

config["tracking"]["preconditioner"] = dict(
    enabled=_on,
    tangent=os.environ.get("PRECOND_TANGENT", "1") not in ("0", "", "false"),
    # PRE_LR IS NOT PORTABLE FROM TUM. The |g|/|g0| stopping criterion is
    # scale-free across frames because each normalises by its own start, but
    # the STEP is lr * P m and P ~ 1/sqrt(E[gg^T]) - so the step scale follows
    # the dataset's gradient magnitudes. Replica is synthetic and noise-free;
    # expect a different optimum and screen it on FULL runs, not on frame-20
    # samples, which reversed twice on TUM.
    lr=float(os.environ.get("PRE_LR", "0.004")),
    beta1=0.9,
    # PRE_BETA2 sets the metric's memory: horizon ~ 1/(1-beta2), so 0.95 is
    # ~20 iterations. That is the knob controlling HOW MUCH THE CARRY IS
    # WORTH. At 0.95 over a 40-iteration frame the carried M has decayed to
    # 0.95^40 = 0.13 by the end - the amortisation is discarded exactly where
    # the budget is tightest and it should matter most. A longer horizon
    # makes each frame lean on the carry instead of re-estimating from
    # scratch, which is the amortisation claim made operational.
    beta2=float(os.environ.get("PRE_BETA2", "0.95")),
    tau=float(os.environ.get("PRE_TAU", "0.05")),
    # PRE_SHRINK: 1.0 = full matrix, 0.0 = diagonal (Adam in the tangent).
    # M is 6x6 symmetric - 21 parameters from rank-1 updates - so at a short
    # budget it is a poor estimate and inverting noisy off-diagonals can cost
    # more than the coupling is worth. This is also the CENTRAL CLAIM made
    # measurable: the whole argument is that the coupling matters, and a
    # sweep here tests it per scene. TUM at 40 iters: full matrix 2.80-4.77cm
    # where Adam was 0/5 all diverged. Replica at 40: the full matrix LOSES
    # to Adam. Those scenes differ in noise, not budget.
    shrink=float(os.environ.get("PRE_SHRINK", "1.0")),
    refactor_every=10,
    max_step_mult=float(os.environ.get("PRE_MAX_STEP", "10.0")),
    # Match the handoff-free optimizer used by the TUM experiments: reset the
    # first moment at the phase boundary, retain M, then refine with the
    # diagonal of that same learned metric. Iteration N is the first diagonal
    # update; zero disables either control.
    restart_at=_restart,
    restart_m=_restart_m,
    diag_after=_diag_after,
    # Per-coordinate rates for the tail only; the full phase keeps PRE_LR.
    # None preserves the shared-rate behaviour of every arm measured so far.
    diag_lr=_diag_lr,
    profile_every=int(os.environ.get("PRE_PROFILE_EVERY", "0")),
    # LR_SERIES=1: print the raw per-frame IT/FRAME, REL-GRAD, K series in
    # the summary. Off by default - lr_ladder.sh sets it for the runs it
    # hands to profiling/lr_proxy_replay.py; nobody else needs it.
    log_series=os.environ.get("LR_SERIES", "0") not in ("0", "", "false", "False"),
    dead_freeze=os.environ.get("PRE_DEAD_FREEZE", "1") not in
        ("0", "", "false", "False"),
    carry=os.environ.get("CARRY", "1") not in ("0", "", "false", "False"),
    # Measured null on TUM at n=5: same time, same iterations, same ATE band,
    # same 4/5. Kept switchable because the ablation should be repeated here -
    # if it is null on two scenes that is a result worth stating.
    transport=os.environ.get("TRANSPORT", "0") not in ("0", "", "false", "False"),
    # AUTO_LR=1 derives the step magnitude from this config's own Adam lrs
    # (sqrt(3 lr_trans^2 + 3 lr_rot^2)) and ignores PRE_LR. That removes the
    # one per-dataset constant the method still had - the metric supplies the
    # direction, which is the contribution, and Adam's already-tuned lrs
    # supply the scale, which never was.
    auto_lr=os.environ.get("AUTO_LR", "0") not in ("0", "", "false", "False"),
    stop_rel_grad=float(os.environ.get("STOP_REL", "0.0667")),
    stop_min_iters=int(os.environ.get("STOP_MIN", "10")),
    stop_ref=os.environ.get("STOP_REF", "running"),
    # THE TUM RESULT'S KNOBS. This config predates all of them, so the
    # combination that reached 6/6 at ~5.5x on fr1_desk could not be run here
    # at all. See the "THE RESULT" section of results/preconditioner_ladder.txt.
    #
    #   STOP_MODE=best   patience on the BEST |g| this frame, not a threshold
    #                    on the current one. |g|/|g_0| is non-monotone and
    #                    plateaus, so a threshold either fires on a noise dip or
    #                    waits ~90 iterations.
    #   FINAL_DENSE=1    one full-resolution iteration AT the stopping moment.
    #                    is_sparse_phase makes only the LAST iteration dense, so
    #                    a frame that stops early never sees a full image.
    #   PRE_HANDOFF=N    preconditioner for the first N iterations, then Adam.
    #                    On TUM this was the difference between 6/6 and 1/3 at
    #                    the same budget - but only at FULL LENGTH; at 250
    #                    frames both arms sat at the accuracy floor and it read
    #                    as neutral twice.
    #
    # NONE OF THIS IS ASSUMED TO TRANSFER. Replica is a different regime -
    # synthetic and noise-free, tracking 40 / mapping 60 so MAPPING DOMINATES
    # and a tracking speedup is diluted, and 2000 frames instead of 592. The
    # ladder already records that PRE_LR is not portable, which is what AUTO_LR
    # exists for.
    stop_mode=os.environ.get("STOP_MODE", "rel"),
    stop_patience=int(os.environ.get("STOP_PATIENCE", "10")),
    stop_improve=float(os.environ.get("STOP_IMPROVE", "0.01")),
    final_dense=os.environ.get("FINAL_DENSE", "0") not in ("0", "", "false", "False"),
    handoff=int(os.environ.get("PRE_HANDOFF", "0")),
    stop_check_every=int(os.environ.get("STOP_EVERY", "5")),
)

# PIXEL_SAMPLE=0 TURNS TILE MASKING OFF. The default is whatever the parent
# set, so this changes nothing unless asked.
#
# WHY THE SWITCH EXISTS. The 2.16x that put pixel_sample in these configs was a
# COMBINED run - tile masking plus a more aggressive early_stop (min_iters=10) -
# and early stopping alone is worth ~7x on SplaTAM, so the sparse contribution
# was never isolated. On MonoGS it HAS been isolated across five cells: 65% of
# iterations masked moved ms/iter by 0.00, at both 6 ms and 12.3 ms per
# iteration, while a 3-5x change in Gaussian count also moved nothing and
# halving the resolution DID. That triangulates the cost onto per-block launch,
# which masking cannot reduce because every block still launches.
#
# So this is the A/B the base config itself asks for: "Re-enable and A/B on its
# own once those are settled."
# ONLINE ACQUISITION-LR TUNER, TAGGED - SAME RULE AND SAME REASON AS THE TUM
# CONFIG.  SplaTAM's normal path tunes the scalar preconditioner lr, not a
# fixed target_step_norm, so an online run must remain distinct from its fixed
# PRE_LR control in both the artifact directory and the suite report.
_online_lr_tune = os.environ.get(
    "ONLINE_LR_TUNE", "0") not in ("0", "", "false", "False")
if _online_lr_tune:
    _online_lr_block = int(os.environ.get("ONLINE_LR_BLOCK", "12"))
    _online_lr_halvings = int(os.environ.get("ONLINE_LR_MAX_HALVINGS", "4"))
    if _online_lr_block <= 0:
        raise ValueError("ONLINE_LR_BLOCK must be > 0")
    if _online_lr_halvings <= 0:
        raise ValueError("ONLINE_LR_MAX_HALVINGS must be > 0")
    config["run_name"] += (
        f"_olrb{_online_lr_block}h{_online_lr_halvings}")

# GRADIENT REUSE, TAGGED - SAME RULE AND SAME REASON AS THE TUM CONFIG.
#
# The flags are read in splatam.py (utils/grad_reuse.py's config_from_env),
# so without this run_name carries no trace of them and a reuse arm lands on
# top of the baseline it exists to be compared against.
#
# READ THE TUM RESULT AS A HYPOTHESIS HERE, NOT AS A SETTING. The warmup of
# 6 was tuned on TUM fr1_desk, where the pose moves roughly twice as fast in
# the first iterations of a frame as it does later, and that is the entire
# reason protecting those iterations recovered the ATE.
#
# Replica's trajectory is smooth and synthetic, and this repo has already
# measured that constant-velocity extrapolation puts the pose essentially AT
# the optimum when a frame opens - a whole 60-iteration budget improving the
# loss by 0.567%. So the early-iteration motion that warmup exists to
# protect may simply not be there, and the gradient may be small enough that
# noise, not rotation, decides how fast it goes stale. Both directions are
# possible and neither is predictable from the TUM numbers.
#
# GRADSTALE=1 measures it directly on this sequence, which is cheaper than
# inferring it from an A/B afterwards.
_gr_env = os.environ.get("GRAD_REUSE")
_gr_cal = int(os.environ.get("GRAD_REUSE_CALIBRATE_FRAMES", "0"))
_gr_hold = int(os.environ.get("GRAD_REUSE_HOLD_FRAMES", "0"))
if _gr_cal < 0:
    raise ValueError("GRAD_REUSE_CALIBRATE_FRAMES must be >= 0")
if _gr_hold < 0:
    raise ValueError("GRAD_REUSE_HOLD_FRAMES must be >= 0")
if (_gr_cal or _gr_hold) and _gr_env in (None, "", "0", "1"):
    raise ValueError(
        "GRAD_REUSE_CALIBRATE_FRAMES/GRAD_REUSE_HOLD_FRAMES require "
        "GRAD_REUSE >= 2"
    )
if _gr_env not in (None, "", "0", "1"):
    config["run_name"] += "_gr%s" % _gr_env
    if _gr_hold:
        config["run_name"] += "h%s" % _gr_hold
    if _gr_cal:
        _gr_margin = os.environ.get("GRAD_REUSE_CALIBRATE_MARGIN", "15")
        _gr_round = os.environ.get("GRAD_REUSE_CALIBRATE_ROUND", "10")
        _gr_stat = os.environ.get("GRAD_REUSE_CALIBRATE_STAT", "min")
        if _gr_stat not in ("min", "median"):
            raise ValueError(
                "GRAD_REUSE_CALIBRATE_STAT must be 'min' or 'median'"
            )
        _gr_short = os.environ.get(
            "GRAD_REUSE_CALIBRATE_SHORT", "1"
        ) not in ("0", "", "false", "False")
        _gr_minwp = os.environ.get("GRAD_REUSE_CALIBRATE_MIN_WARMUP_PERIODS", "3")
        config["run_name"] += (
            f"cal{_gr_cal}m{_gr_margin}r{_gr_round}so{_gr_stat}"
            + ("sh" if _gr_short else "nosh")
            + ("" if _gr_minwp == "3" else f"mwp{_gr_minwp}")
        )
    else:
        _gr_w = os.environ.get("GRAD_REUSE_WARMUP")
        if _gr_w is not None:
            config["run_name"] += "w%s" % _gr_w
        _gr_c = os.environ.get("GRAD_REUSE_COOLDOWN")
        if _gr_c is not None:
            config["run_name"] += "c%s" % _gr_c
    if os.environ.get("GRAD_REUSE_NOACC", "0") not in ("0", "", "false", "False"):
        config["run_name"] += "na"
    if os.environ.get("GRAD_REUSE_FREEZE_M", "0") not in ("0", "", "false", "False"):
        config["run_name"] += "fm"
    if os.environ.get("GRAD_REUSE_ADAPTIVE", "0") not in ("0", "", "false", "False"):
        _gr_trust = os.environ.get("GRAD_REUSE_TRUST_COS", "0.85")
        _gr_check = os.environ.get("GRAD_REUSE_CHECK_EVERY", "4")
        config["run_name"] += (
            "adt%sq%s" % (_gr_trust.replace(".", ""), _gr_check)
        )

# GRADSTALE=1: the staleness probe, so the measurement that justified
# gradient reuse can be repeated on THIS sequence instead of assumed.
# Graphs forced off - the probe runs extra forward+backward passes inside
# the tracking loop, which launch into a non-capturing stream under a
# capture.
if os.environ.get("GRADSTALE", "0") not in ("0", "", "false", "False"):
    config["tracking"]["grad_staleness"] = {
        "enabled": True,
        # Earlier than TUM's 50/150/250: Replica room0 runs 500 frames in
        # this suite, so these sample a thin, a maturing and a mature map
        # without falling off the end of the sequence.
        "frames": [50, 150, 300],
        # Replica tracks at ~29 iterations a frame here, so the deepest
        # offset (15 + 4) still lands inside a real frame.
        "starts": [5, 15],
        "offsets": [1, 2, 4],
        "usable_cos": 0.95,
        "out_path": "grad_staleness_replica.jsonl",
    }
    config["tracking"]["cuda_graph"]["enabled"] = False
    config["tracking"].setdefault("iteration_graph", {})["enabled"] = False
    config["run_name"] += "_gstale"
_pixel_sample_env = os.environ.get("PIXEL_SAMPLE")
if _pixel_sample_env is not None:
    _ps_on = _pixel_sample_env not in ("0", "", "false", "False")
    config["tracking"]["pixel_sample"]["enabled"] = _ps_on
    # TAGGED, because run_name keys the output directory and params.npz. An
    # untagged sparse-off arm would overwrite the sparse-on run it exists to be
    # compared against - the bug class this repo has hit nineteen times.
    config["run_name"] += "_nops" if not _ps_on else "_ps"
if os.environ.get("BINCAP", "1") in ("0", "", "false", "False"):
    # TAGGED: run_name keys the output directory, and a binning-off arm
    # must not overwrite the binning-on run it exists to be compared with.
    config["run_name"] += "_nobin"
