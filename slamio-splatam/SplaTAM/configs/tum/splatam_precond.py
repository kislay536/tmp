# FULL-MATRIX POSE PRECONDITIONER, Stage 0. And its matched Adam control.
#
#   PRECOND=1 ITERS=90 python scripts/splatam.py configs/tum/splatam_precond.py
#   PRECOND=0 ITERS=90 python scripts/splatam.py configs/tum/splatam_precond.py
#
# ONE FILE FOR BOTH ARMS, deliberately. The preconditioner cannot run with
# either CUDA graph enabled - both capture `optimizer.step()`, which it
# replaces - so its arm must have them off. If the Adam control kept them on,
# every wall-time comparison would be measuring the graph, not the method, and
# the ladder has already produced one perfectly separated fake result from an
# uncontrolled difference between arms (see the A/B counterbalancing note).
# PRECOND=0 gives Adam with the graphs off too, so the only thing that differs
# between the arms is the update rule.
#
# This means the ABSOLUTE times here are slower than the dose-curve runs, which
# had cuda_graph on. Do not compare across the two sets. The number that
# transfers is the RATIO between PRECOND=1 and PRECOND=0 at the same ITERS.
#
# WHAT STAGE 0 IS, AND WHAT IT IS NOT. n = 7 on the existing (unnormalised
# quaternion, translation) coordinates. Not the principled version: the
# quaternion's radial direction is an exact null direction of the loss, and
# there is no SE(3) adjoint in these coordinates, so the cross-frame carry
# cannot be transported and runs with transport=None (i.e. carried untransported
# - the frame-to-frame rotation of the metric is simply not applied). Stage 1
# moves to the 6-D tangent where the carry becomes M <- A^-T M A^-1 and the
# amortisation claim is actually available.
#
# So a null result here does NOT kill the method; it kills the cheap version of
# it. A positive result is the go-ahead for the tangent surgery.
#
# WHAT TO MEASURE. ATE and PSNR say whether accuracy is preserved. Tracking
# ms/iteration says what the 7x7 solve costs per step - it should be nearly
# nothing against an ~11 ms render, and if it is not, the refactor_every
# schedule is wrong. The claim being tested is that the preconditioner holds
# accuracy at a budget where Adam does not.
#
# AND REMEMBER THE BIMODALITY. At 90 iterations stock Adam held the basin on
# 2 of 5 replicates (3.65, 3.70 against 11.53, 16.62, 17.09), cleanly separated
# with nothing in between. A single run of either arm here is one draw from a
# coin flip and settles nothing. The first run is a SMOKE TEST - does it run,
# does it diverge, what does it cost per iteration - and the comparison needs
# REPS=5 on both arms afterwards.
from configs.tum.splatam_adam_dose import config as _base
import copy
import os

_on = os.environ.get("PRECOND", "1") not in ("0", "", "false", "False")

config = copy.deepcopy(_base)
_iters = config["tracking"]["num_iters"]
_frames = config["data"]["num_frames"]
# MAP_ITERS, for the mapping-side MapTrace (utils/map_iter_trace.py): the
# base config's mapping.num_iters (30) was never meant to be varied, unlike
# ITERS on the tracking side. Unset reproduces every existing run exactly.
if "MAP_ITERS" in os.environ:
    config["mapping"]["num_iters"] = int(os.environ["MAP_ITERS"])
# RUN_REP keeps each replicate's params.npz. Without it every rep writes to
# one directory and overwrites the last, so only the final run's trajectory
# survives - which is why the 5/5 sweep could not be re-examined with
# ate_vs_length.py when a later full-length run disagreed with it. The
# trajectories were already paid for and had been thrown away.
_rep = os.environ.get("RUN_REP", "")
_shrink = float(os.environ.get("PRE_SHRINK", "1.0"))
_samples = os.environ.get("PRE_SAMPLES", "0") not in ("0", "", "false", "False")
# C1. PRE_RESTART=N zeroes the first moment at within-frame iteration N.
_restart = int(os.environ.get("PRE_RESTART", "0"))
_restart_m = os.environ.get("PRE_RESTART_M", "off")
# COMMIT_AT_LOSS=1: commit the pose the best loss was EVALUATED at, instead of
# the one the optimiser stepped to immediately afterwards.
#
# THIS IS A CORRECTNESS FIX, NOT A TUNING KNOB. get_loss runs at the top of a
# tracking iteration, so the loss describes the PRE-step pose; the candidate
# update then reads the pose AFTER the step. Every frame therefore commits a
# pose one optimiser step past its own best, and the error is the size of that
# step.
#
# IT SURFACES AS MAP QUALITY, NOT TRAJECTORY ERROR, because ATE is aligned over
# the whole run so a per-frame offset partly cancels, while every rendered
# frame carries the full misalignment. Measured full length on TUM fr1:
#   handoff      Adam tail 0.49x ref   ATE 3.86   PSNR 19.00
#   handoff-free tail 0.04-0.08x ref   ATE 3.38   PSNR 21.53
# ~6-8x more commit error in the arm with the 2.5 dB deficit.
#
# Default OFF. It changes every committed pose in every arm, and both ladders'
# numbers were recorded with the current behaviour - the PRE_MAX_STEP and
# STOP_ANCHOR precedent: add the flag, measure, then flip. Present in the eager
# path, the sync-free path and es_signals' batched path, and in all three
# models; only SplaTAM is wired here.
# LEVELS, because the pairing fix alone turned out to be HALF a fix:
#   0  original. Commit pose_{k+1} scored by loss(pose_k).
#   1  correct pairing. Commit pose_k scored by loss(pose_k). Removes the
#      overshoot bias - but drops the FINAL pose from consideration entirely,
#      so on a monotonically converging frame it throws away a real step.
#   2  correct pairing AND score the final pose, with one extra forward per
#      frame (~1 render against ~29). Removes both biases.
#
# MEASURED, why level 1 is not enough. Full length, TUM fr1:
#   handoff arm (Adam tail 0.49x ref, overshoots)
#       ATE 3.86 -> 3.31, PSNR 19.00 -> 21.08   level 1 HELPS
#   handoff-free + restart (tail 0.19x, near-monotone)
#       ATE 3.13/3.49 -> 4.65, PSNR 21.46 -> 21.47   level 1 HURTS
# The restart arm's split is the signature: every MAP metric its best of any
# run and only ATE bad, which is what a systematic one-step lag does - it
# biases the trajectory against ground truth while leaving the map
# self-consistent with the poses it was built from.
_commit_at_loss = int(os.environ.get("COMMIT_AT_LOSS", "0"))
config["tracking"].setdefault("early_stop", {}).setdefault(
    "batched", {})["pose_at_loss"] = _commit_at_loss
# PRE_EXCURSION_TRACE is forensic instrumentation, not an optimisation arm.
# "1" selects a run-name-derived path; any other non-empty value is used as
# the path verbatim. It synchronises every tracking iteration and therefore
# invalidates timing, but does not change the optimiser arithmetic. A bounded
# rolling history is written only around the first trigger.
_trace_req = os.environ.get("PRE_EXCURSION_TRACE", "").strip()
# C2. PRE_RAMP="from:to", or "N" for a hard switch at N (from == to).
# Parsed here rather than in two places so the run_name tag below and the
# constructor argument can never disagree about what was asked for.
_ramp = os.environ.get("PRE_RAMP", "").strip()
if not _ramp or _ramp == "0":
    _ramp_from = _ramp_to = 0
elif ":" in _ramp:
    _ramp_from, _ramp_to = (int(x) for x in _ramp.split(":", 1))
else:
    _ramp_from = _ramp_to = int(_ramp)
# Full-matrix acquisition followed by diagonal second-moment refinement. This
# is deliberately separate from PRE_RAMP: ramp targets constant-norm gradient
# descent, while this keeps Adam-like learned per-coordinate scales from the
# same M. Iteration N is the first diagonal step; 0 disables.
_diag_after = int(os.environ.get("PRE_DIAG_AFTER", "0"))
if _diag_after < 0:
    raise ValueError("PRE_DIAG_AFTER must be >= 0")
if _diag_after and _ramp_to:
    raise ValueError("PRE_DIAG_AFTER and PRE_RAMP are alternative tail rules")

# THE CALIBRATED ALTERNATIVE TO CHOOSING _diag_after BY HAND.
# Every full-matrix step is a proposal; the next render scores it against the
# loss it was taken from; after `patience` consecutive non-improvements the
# pose rolls back and the diagonal replaces that proposal. With
# PRE_DIAG_CALIB=N the class takes one sample per frame, installs the median
# as diag_after after N frames, and clears adaptive_diag - so the rest of the
# run is an ordinary fixed-D arm with a value nobody chose.
#
# THE POINT OF RUNNING IT HERE. SplaTAM/TUM fr1 is where the hand-set value is
# best established - diag_after=20 at ~30 iterations, 5/5 against Adam's 0/5
# at n>=5, p~0.008. So it is the one sequence where "did the calibration find
# the good value" has a trustworthy answer, and it is minutes per run.
_adaptive_diag = os.environ.get("PRE_ADAPTIVE_DIAG", "0") not in (
    "0", "", "false", "False")
_diag_patience = int(os.environ.get("PRE_DIAG_PATIENCE", "1"))
_diag_calib = int(os.environ.get("PRE_DIAG_CALIB", "0"))
_diag_min = int(os.environ.get("PRE_DIAG_MIN", "0"))
if _adaptive_diag and _diag_after:
    raise ValueError(
        "PRE_ADAPTIVE_DIAG and PRE_DIAG_AFTER are alternative tail rules - "
        "the adaptive transition exists to REPLACE the hand-set one")
if _adaptive_diag and _ramp_to:
    raise ValueError("PRE_ADAPTIVE_DIAG and PRE_RAMP are alternative tail rules")
# SHADOW CALIBRATION: MEASURE THE TRANSITION WITHOUT APPLYING IT.
#
# WHY IT EXISTS. The applying calibration damages the trajectory it calibrates
# on, and the damage is (how wrong the applied value is) x (how many frames it
# is applied for). Measured on this arm: floored to 20 with 24 calibration
# frames scored 5.39 cm; the same thing with 4 frames scored 3.61, inside the
# fixed arm's 3.12-3.67 band. Nothing else differed. The 24 frames at the
# start are where the map is established and errors are least recoverable.
#
# WHAT SHADOW DOES INSTEAD. Run the accepted PRE_DIAG_AFTER for real, and
# separately record the iteration at which the adaptive rule WOULD have
# switched. When the samples are in, install their median. Same evidence, no
# trajectory cost - so the sample count can be chosen on statistical grounds
# instead of as a damage budget.
#
# THE ONE LIMITATION, AND IT IS RIGHT-CENSORING RATHER THAN BIAS. The
# observation is only valid while the frame is still taking FULL steps, i.e.
# below PRE_DIAG_AFTER. A frame whose would-be switch lies beyond that is
# recorded as censored at the safe value, exactly as a frame that converges
# before switching already is. Not binding in practice: the rule has selected
# 2-4 in every configuration measured.
#
#   PRE_DIAG_SHADOW=1 PRE_DIAG_AFTER=20 PRE_DIAG_CALIB=24
#
_diag_shadow = os.environ.get("PRE_DIAG_SHADOW", "0") not in (
    "0", "", "false", "False")
if _diag_shadow:
    if _adaptive_diag:
        raise ValueError(
            "PRE_DIAG_SHADOW and PRE_ADAPTIVE_DIAG are alternatives - shadow "
            "measures the transition while a fixed one runs, adaptive applies "
            "what it measures")
    if not _diag_after:
        raise ValueError(
            "PRE_DIAG_SHADOW needs PRE_DIAG_AFTER: it is the transition that "
            "actually runs while the adaptive rule is being measured")
    if not _diag_calib:
        raise ValueError("PRE_DIAG_SHADOW requires PRE_DIAG_CALIB")

if _diag_calib and not (_adaptive_diag or _diag_shadow):
    # BOTH calibrating modes consume it. This read _adaptive_diag alone and
    # rejected every shadow run at import, because shadow deliberately leaves
    # adaptive_diag off - the whole point is that the class runs the fixed
    # transition while the rule is only watched.
    raise ValueError(
        "PRE_DIAG_CALIB requires PRE_ADAPTIVE_DIAG=1 or PRE_DIAG_SHADOW=1")

# ---------------------------------------------------------------------------
# PRE_DIAG_LR - the diagonal tail runs on this model's OWN Adam rates.
#
# The full phase and the diagonal tail SHARE ONE RATE by default: the class
# fills diag_lr with self.lr when nothing is passed. That is exact only where
# the config's rotation and translation rates are EQUAL, and SplaTAM/TUM is the
# one scene in this repo where they are - both 0.002. Everywhere else a single
# shared rate cannot represent the split:
#
#   SplaTAM  TUM      rot 0.002   trans 0.002    1x   shared rate is exact
#   SplaTAM  Replica  rot 0.0004  trans 0.002    5x   shared rate is a blend
#   SplaTAM  ScanNet  rot 0.0005  trans 0.0005   1x
#   GSLAM    TUM      rot 0.002   trans 0.01     5x   wired since its port
#   MonoGS   every    rot 0.003   trans 0.001    3x   wired since its port
#
# GSLAM (tracker.py) and MonoGS (slam_frontend.py) both inject this from their
# configs already. SplaTAM never did - because on TUM it would have changed
# NOTHING, which is exactly why the omission survived and why it becomes a
# confound the moment a port moves to a scene whose two rates differ.
#
# THE ORDER IS [trans]*3 + [rot]*3, AND THAT IS NOT A FREE CHOICE.
# quat_trans_grad_to_tangent returns cat([g_rho, g_theta]) and
# se3_exp_capturable reads `rho, theta = xi[:3], xi[3:]`, so the tangent is
# translation-first. GSLAM uses the same shared helpers and the same order;
# MonoGS is rotation-first because it builds its own tangent. Swapping them
# silently applies the rotation rate to translation - on Replica a factor of
# five in the wrong direction, and it would read as a scene effect.
#
# OFF BY DEFAULT, AND TAGGED WHEN ON. Enabling it unannounced would change
# every existing arm running PRE_DIAG_AFTER=20 - the production incumbent
# included - and silently re-point a validated reference. Opt-in plus a run
# name tag keeps it one variable against the arms already in the ladder.
#
#   PRE_DIAG_LR=1        derive [cam_trans]*3 + [cam_unnorm_rots]*3
#   PRE_DIAG_LR=t,r      explicit pair, same trans-then-rot order
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
    if not (_diag_after or _adaptive_diag):
        # The class raises this itself, but only at construction and without
        # naming the env var that caused it.
        raise ValueError(
            "PRE_DIAG_LR needs a diagonal tail to apply to: set "
            "PRE_DIAG_AFTER=N or PRE_ADAPTIVE_DIAG=1. With the full matrix "
            "running the whole frame there is no tail for these rates to "
            "reach - which is why the 6/6 lr0.004 reference, and the Replica "
            "k port that copies it, are unaffected by this flag.")
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

if _adaptive_diag and os.environ.get("GRAPH", "0") not in (
        "0", "", "false", "False"):
    # TWO REASONS, either of which is sufficient. The decision reads the loss
    # to the host every acquisition iteration, which is a synchronisation; and
    # a rejected iteration SKIPS _precond_update entirely, so the captured
    # step would not be replayed on the iteration the graph expects it. The
    # readback disappears once calibration installs a fixed diag_after, so
    # GRAPH=1 is usable for the arms that follow - just not for this one.
    raise ValueError(
        "PRE_ADAPTIVE_DIAG cannot run with GRAPH=1: the transition reads the "
        "loss each acquisition iteration and skips the captured step on a "
        "rejection. Run the calibration arm graph-off; the graph changes "
        "execution and not arithmetic, so its ATE stays comparable.")
config["run_name"] = (f"freiburg1_desk_precond{'on' if _on else 'off'}"
                      f"_i{_iters}_f{_frames}")
# PRE_SHRINK goes in run_name for exactly the reason RUN_REP does: the shell
# tags gamma into the LOG name, but SplaTAM keys its output directory and
# params.npz on run_name, so without this the five gamma arms all write to one
# directory and only the last survives - and the sweep's trajectories, already
# paid for, could not be re-examined with ate_vs_length.py.
if _on and _shrink != 1.0:
    config["run_name"] += f"_s{_shrink:g}"
# SAME REASON, THIRD VARIABLE. f1d31ea fixed this for RUN_REP and 403b060 for
# PRE_SHRINK, each time after an already-paid sweep had overwritten its own
# trajectories. Tag the arm before running it, not after losing it.
if _on and _samples:
    config["run_name"] += "_smp"
_ho = os.environ.get("PRE_HANDOFF", "0")
if _on and _ho not in ("0", ""):
    config["run_name"] += f"_ho{_ho}"
_sr = os.environ.get("STOP_REL", "0")
if _on and _sr not in ("0", "0.0", ""):
    _sm = os.environ.get("STOP_MODE", "rel")
    if _sm == "best":
        config["run_name"] += f"_best{os.environ.get('STOP_PATIENCE', '10')}"
    else:
        config["run_name"] += f"_sr{_sr}"
    if os.environ.get("FINAL_DENSE", "0") not in ("0", "", "false", "False"):
        config["run_name"] += "fd"
    if os.environ.get("STOP_REF", "frame") != "frame":
        config["run_name"] += "ref"
    _mn = os.environ.get("STOP_MIN", "10")
    if _mn != "10":
        config["run_name"] += f"m{_mn}"
if _on and os.environ.get("PRE_M0", "") == "iso":
    config["run_name"] += "_m0iso"
if _on and os.environ.get("BFGS", "0") not in ("0", "", "false", "False"):
    config["run_name"] += "_bfgs"
if _on and os.environ.get("PRE_BETA1", "0.9") != "0.9":
    config["run_name"] += f"_b1{os.environ.get('PRE_BETA1')}"
# INSTANCE EIGHTEEN, AND IT HAD ALREADY DESTROYED TWO RUNS BEFORE IT WAS
# NOTICED. PRE_BETA1 was tagged and PRE_BETA2 was not, though beta2 is the more
# consequential of the two - it sets the metric's entire memory. A PRE_BETA2
# arm therefore wrote to the same directory as the default-beta2 run it was
# being compared against, and overwrote its params.npz; the b2=0.9 pair in the
# it20-39 investigation lost both baselines that way. Only the logs survived.
if _on and os.environ.get("PRE_BETA2", "0.95") != "0.95":
    config["run_name"] += f"_b2{os.environ.get('PRE_BETA2')}"
if _on and os.environ.get("CARRY", "1") in ("0", "", "false", "False"):
    config["run_name"] += "_nocarry"
# INSTANCE SEVENTEEN, AND IT IS ALREADY LIVE. run_adam_dose.sh tags PRE_MAX_STEP
# and STOP_ANCHOR into the LOG name (_MTAG/_ATAG) but nothing tagged them into
# run_name - so every arm of the stability sweep that differs ONLY in those two
# knobs (control, maxstep, anchor, combo) shares one output directory and one
# params.npz. The logs are distinct, so the ATEs in the ladder are real; what is
# gone is the ability to re-examine any of those trajectories with
# ate_vs_length.py, because each arm overwrote the last.
#
# Same defect, same cause and the same fix as f1d31ea (RUN_REP), 403b060
# (PRE_SHRINK) and b8d00cd (the shrink sweep's tag). Tag the arm before running
# it, not after losing it.
_msv = os.environ.get("PRE_MAX_STEP", "10.0")
if _on and _msv not in ("10.0", "10", ""):
    config["run_name"] += f"_ms{_msv}"
if _on and os.environ.get("STOP_ANCHOR", "min") != "min":
    config["run_name"] += f"_anch{os.environ.get('STOP_ANCHOR')}"
# C1 / C2, tagged at the same time they are added rather than after a sweep has
# already overwritten itself.
if _on and _restart:
    config["run_name"] += f"_rst{_restart}"
    if _restart_m != "off":
        config["run_name"] += _restart_m
# TAGGED, because it changes what every frame commits - the most consequential
# flag added in this file. Not gated on _on: it applies to the Adam control too.
# _cal stays level 1 so the runs already recorded under that tag keep their
# meaning; level 2 gets its own.
if _commit_at_loss == 1:
    config["run_name"] += "_cal"
elif _commit_at_loss >= 2:
    config["run_name"] += "_cal2"
# Tagged only when DISABLED: enabled is the new default and a no-op on healthy
# runs, so tagging it would rename every arm for nothing.
if _on and os.environ.get("PRE_DEAD_FREEZE", "1") in ("0", "", "false", "False"):
    config["run_name"] += "_nodf"
_dm = os.environ.get("DRIFT_MULT", "0")
if _on and _dm not in ("0", "0.0", ""):
    config["run_name"] += f"_drift{_dm}"
if _on and _ramp_to:
    config["run_name"] += (f"_ramp{_ramp_from}" if _ramp_from == _ramp_to
                           else f"_ramp{_ramp_from}-{_ramp_to}")
if _on and _diag_after:
    config["run_name"] += f"_diag{_diag_after}"
if _on and _diag_shadow:
    config["run_name"] += f"_shdiag{_diag_patience}c{_diag_calib}"
    if _diag_min:
        config["run_name"] += f"m{_diag_min}"
if _on and _adaptive_diag:
    config["run_name"] += f"_adiag{_diag_patience}"
    if _diag_calib:
        config["run_name"] += f"c{_diag_calib}"
    if _diag_min:
        config["run_name"] += f"m{_diag_min}"
# TAGGED, LIKE EVERY OTHER ARM-DEFINING KNOB. Four separate arms in this
# record did nothing at all because their env var never reached the log name
# and the runner skipped them on an existing file, or averaged two different
# configurations into one row.
if os.environ.get(
        "WCONV_ENERGY_PHASE_RELATIVE", "1"
        ) in ("0", "", "false", "False"):
    # Frame-relative energy changes only incumbent_energy, but still changes
    # where a frame may stop and therefore must own a distinct artifact path.
    config["run_name"] += "_eprF"
_anomaly_release_after = int(os.environ.get(
    "WCONV_ANOMALY_RELEASE_AFTER", "0"
))
if _on and _anomaly_release_after > 0:
    # The release changes which formerly barred frame is committed.  Keep its
    # trajectory artifacts separate even when REP was accidentally reused.
    config["run_name"] += f"_anrel{_anomaly_release_after}"
if _on and _diag_lr_tag:
    config["run_name"] += _diag_lr_tag
if _rep:
    config["run_name"] += f"_r{_rep}"

# GRAPH=1 enables the tracking-step CUDA graph, for BOTH arms.
#
# This used to be impossible: the graph captures optimizer.step(), which the
# preconditioner REPLACES rather than feeds, and splatam.py refused the
# combination outright. TrackingStepGraph.step() now takes a callable, so the
# preconditioner's own update is captured instead - which required rewriting
# se3_exp and mat_to_quat branch-free (both read a tensor back to the host and
# branched on it, and a capture freezes whichever branch it records) and moving
# the bias correction to a device-side counter.
#
# Default OFF so every result measured so far stays reproducible. When it is
# on it must be on for BOTH arms: the baseline dose runs have it, and comparing
# a graph-off preconditioner against a graph-on Adam measures the graph, not
# the method. That confound is why the fixed-200 arm reads 1839.5 s against the
# baseline's 1630.3 s at identical iteration counts.
_graph = os.environ.get("GRAPH", "0") not in ("0", "", "false", "False")
config["tracking"]["cuda_graph"]["enabled"] = _graph
if _graph:
    config["run_name"] += "_g"
# ITERGRAPH=1 CAPTURES THE WHOLE TRACKING ITERATION. This was hardcoded off
# here with the note that it "is still incompatible", and that note was STALE:
# splatam.py no longer refuses the combination, it prints
#
#   "[Preconditioner] iteration_graph is ON with the preconditioner. The
#    update is capture-safe but this combination has not been validated
#    end-to-end - check the capture summary and that ATE matches a graph-off
#    run."
#
# which is a warning, not a guard. configs/replica/splatam_precond.py has
# exposed it as an env flag for some time; this brings TUM in line so one
# switch means the same thing on both scenes.
#
# IT IS THE GRAPH WORTH HAVING. splatam.py's own comment puts the iteration
# graph at 2.71x on the SplaTAM ladder AGAINST THE STEP GRAPH'S MEASURED NET
# LOSS - so GRAPH=1 above is the one that has been shown to cost time, and
# this is the one that has been shown to save it. They are alternatives: with
# the iteration graph on, the step graph is redundant because the update is
# already inside the outer capture.
#
# TWO PRECONDITIONS ARE ALREADY MET ON THIS ARM and are worth knowing because
# breaking either makes the capture silently wrong rather than failing:
#   binning_capacity must be enabled, so the rasterizer does not read
#     num_rendered back to the host mid-capture. It is on below.
#   the tile mask must be CONSTANT within a frame, so a captured iteration
#     cannot freeze one phase of a changing mask. SplaTAM's pixel_sample uses
#     the always-sparse [0.0, 1.0] window, which satisfies this by
#     construction - the narrow window MonoGS ships would not.
_itergraph = os.environ.get("ITERGRAPH", "0") not in ("0", "", "false", "False")
config["tracking"].setdefault("iteration_graph", {})["enabled"] = _itergraph
config["tracking"]["iteration_graph"].setdefault("warmup_iters", 5)
if _itergraph:
    config["run_name"] += "_ig"

# ADMAP=0 TURNS ADAPTIVE MAPPING OFF (the chain's own default - the base
# this file inherits from leaves it off, unset ADMAP changes nothing). ADMAP=1
# turns it ON WITH SELF-CALIBRATING depth/colour/new-points references
# instead of the chain's hand-set depth_error_ref=0.01/color_error_ref=0.05/
# n_new_pts_ref=200, none of which transfer across scenes (on Replica the
# depth/colour pair reads every frame as converged and collapses the budget;
# n_new_pts_ref=200 is an unfitted guess, the same failure mode that lost
# GSLAM tracking on ScanNet at frame ~1000 before it was calibrated there).
# calibration_keyframes runs the first skip+N mapped frames at the full
# budget - what the arm does with the mechanism off, so the window costs
# nothing - fits depth_error_ref/color_error_ref to the p90 of THIS scene's
# own errors and freezes them: the shape MonoGS's admap cells use
# (calibration_keyframes: 24). calibrate_new_pts_frames/calibrate_new_pts_skip
# do the same for n_new_pts_ref, on their OWN separate window - the two
# calibrations are decoupled in AdaptiveMapper (see utils/adaptive_mapper.py)
# because a caller that never supplies depth/colour error (GSLAM) must not
# block n_new_pts_ref's fit on a counter that never advances. SplaTAM maps
# every frame, so N=24 plus the 8 discarded warm-up frames is the first 32
# frames, for both windows. AM_CALIB_KF, AM_CALIB_SKIP, AM_CALIB_NP,
# AM_CALIB_NP_SKIP, AM_MIN_ITERS and AM_MAX_ITERS override.
#
# REGRESSION, FIXED HERE (depth/colour half only). This calibration block
# (originally 8dee1e2) was deleted by 2c0006c in the same change that added
# the ADMAP=0 hook, leaving ADMAP=1 as a bare enable with no calibration
# while a comment kept claiming otherwise - the exact uncalibrated-thresholds
# failure mode the block exists to avoid. Restored, merged with the disable
# hook so one variable does both directions. The new-points half never
# existed here before at all: splatam.py used to feed unseen_ratio into both
# signals, so n_new_pts_ref was dead regardless of calibration - fixed
# separately in splatam.py/scripts (decoupled the two signals) so this
# reference is now live, calibrated or not.
_admap_env = os.environ.get("ADMAP")
if _admap_env is not None:
    _admap_on = _admap_env not in ("0", "", "false", "False")
    config["adaptive_mapping"]["enabled"] = _admap_on
    if not _admap_on:
        config["run_name"] += "_noadmap"
    else:
        _am_cal = int(os.environ.get("AM_CALIB_KF", "24"))
        _am_skip = int(os.environ.get("AM_CALIB_SKIP", "8"))
        _am_np_cal = int(os.environ.get("AM_CALIB_NP", "24"))
        _am_np_skip = int(os.environ.get("AM_CALIB_NP_SKIP", "8"))
        config["adaptive_mapping"].update(
            enabled=True,
            min_iters=int(os.environ.get(
                "AM_MIN_ITERS", str(config["adaptive_mapping"]["min_iters"]))),
            max_iters=int(os.environ.get(
                "AM_MAX_ITERS", str(config["mapping"]["num_iters"]))),
            calibration_keyframes=_am_cal,
            calibration_skip=_am_skip,
            calibrate_new_pts_frames=_am_np_cal,
            calibrate_new_pts_skip=_am_np_skip,
        )
        # TAGGED, because it changes the map every later frame is tracked
        # against and the shell does not tag it: two arms differing only in
        # ADMAP would otherwise write one params.npz.
        config["run_name"] += f"_admapcal{_am_cal}"
        if _am_skip != 8:
            config["run_name"] += f"s{_am_skip}"
        _amin = config["adaptive_mapping"]["min_iters"]
        if _amin != 22:
            config["run_name"] += f"_ammin{_amin}"
        config["run_name"] += f"_npcal{_am_np_cal}"
        if _am_np_skip != 8:
            config["run_name"] += f"s{_am_np_skip}"

# BINNING HEADROOM, for both arms. A 40-iteration run overflowed at frame 9
# ("the first overflow already produced a wrong render"), which corrupts one
# frame's gradient early enough to knock the tracker out of the basin on its
# own. A rep that overflowed is not a measurement of the optimiser - it is a
# measurement of a broken render - and at a reduced budget there is no slack to
# recover from one.
#
# margin=1.25 was calibrated on the Adam dose runs at 90-200 iterations, where
# the observed max is large and stable. At 40 iterations the warmup sees far
# fewer instances, so a RELATIVE margin off a small observed_max leaves too
# little absolute headroom - which is exactly what min_capacity is for.
# Raising both for both arms keeps the comparison matched.
# BIN_MARGIN / BIN_MIN, defaulting to the BASELINE's settings.
#
# These were raised to 2.0 / 1.2M after a 40-iteration run overflowed at frame 9
# ("the first overflow already produced a wrong render"). That was real, but it
# applied to a short-budget regime whose warmup saw very few instances. At the
# 200-iteration caps used since, every run reports 0 overflow frames while
# carrying capacity 1.86M against ~908k observed - 105% headroom.
#
# Worse than wasteful: dose_i200_f-1, the run everything is compared against,
# uses margin 1.25 and no min_capacity. Leaving the preconditioner arms at 2.0
# is an uncontrolled difference between the arms, in the direction that
# penalises the one being measured. Matching the baseline is the correct
# default; raise it again if the ovf column stops reading 0.
config["tracking"]["binning_capacity"] = dict(
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

    # 2.0, NOT the baseline's 1.25. Matching the baseline was the intent,
    # but 1.25 is measurably UNSAFE for this tracker: it overflowed at
    # frame 303, and BinningCapacity's own warning is that the first
    # overflow already produced a wrong render. The preconditioner moves
    # the pose enough WITHIN a frame to bring new Gaussians into view
    # after capacity is fixed from the 3 warmup iterations, which Adam
    # does not. The remaining mismatch against the baseline has to be
    # closed by re-running the BASELINE at 2.0, not by using a margin
    # that corrupts a render.
    margin=float(os.environ.get("BIN_MARGIN", "2.0")),
    min_capacity=int(os.environ.get("BIN_MIN", "0")),
    warmup_iters=3,
)

# ADAPTIVE BUDGET. ES=1 turns early stopping back on and makes ITERS a CAP
# rather than an exact count.
#
# WHY, from the full-length prefix curves: EVERY run bumps at frame ~325,
# including the 200-iteration baseline (3.83 -> 4.24). It is a hard section of
# the sequence, not a property of any tracker. At a fixed 40 iterations three
# of five runs absorbed it and recovered (finishing 3.11 / 4.50 / 5.12 against
# the baseline's 3.58 - run5 BEAT the stock config at a quarter of the budget)
# and two diverged past recovery (35.51, 104.22).
#
# So the failure is localised, and a uniformly larger budget pays for it on all
# 592 frames to fix ~25 of them. Spending iterations where they are needed is
# the whole point of the early-stop machinery already on this branch, and a
# fixed budget was only ever a measurement device for the dose curve.
#
# min_iters is the knob that matters: the base config's 70 is a floor ABOVE the
# 40 that already works on easy frames, so it must come down or early stopping
# cannot save anything.
if os.environ.get("ES", "0") not in ("0", "", "false", "False"):
    config["tracking"]["early_stop"]["enabled"] = True
    config["tracking"]["early_stop"]["min_iters"] = int(os.environ.get("ES_MIN", "25"))
    config["tracking"]["early_stop"]["warmup_min_iters"] = int(os.environ.get("ES_MIN", "25"))
    # RETUNE: off by default, but now reachable.
    #
    # WHY IT WAS OFF: the probe caps frames at retune_probe_max_iters_frac of
    # the budget, and that perturbation has to be absent from a hard section.
    # That is a property of the FRAC, not of retuning, so the frac is exposed
    # too and defaults to 1.0 whenever retuning is on - no truncation at all,
    # only the loss of early stopping for retune_frames frames.
    #
    # WHY IT IS WORTH REACHING FOR. loss_eps is a RELATIVE threshold
    # (EarlyStop._rel divides by the frame's starting loss), so it transfers
    # across frames - but NOT across optimisers, because it is really a
    # statement about step size. Measured: 1e-4 fires at ~49 iters/frame on the
    # preconditioner, whose step is 0.18-0.26x Adam's, and NEVER fires after a
    # handoff to Adam - 100 frames at 200/200. A hand-set threshold cannot be
    # right for both phases. The retuner measures the scale in-run instead,
    # which is exactly the problem it was written for.
    #
    # AND THE INHERITED FLOOR IS A TRAP. retune_min_iters_floor comes in at 70,
    # far above ES_MIN=25, so the first retune would RAISE min_iters to 70 and
    # undo the low-budget operating point the arm exists to test. It defaults
    # to ES_MIN here instead.
    config["tracking"]["early_stop"]["retune_every"] = int(
        os.environ.get("ES_RETUNE", "0"))
    if config["tracking"]["early_stop"]["retune_every"] > 0:
        config["tracking"]["early_stop"]["retune_probe_max_iters_frac"] = float(
            os.environ.get("ES_RETUNE_FRAC", "1.0"))
        config["tracking"]["early_stop"]["retune_min_iters_floor"] = int(
            os.environ.get("ES_RETUNE_FLOOR", os.environ.get("ES_MIN", "25")))
        config["tracking"]["early_stop"]["retune_frames"] = int(
            os.environ.get("ES_RETUNE_FRAMES", "15"))

    # THE THRESHOLDS THE CHAIN INHERITS ARE TOO LOOSE TO ALLOCATE ANYTHING.
    # splatam_es_graph_eps_* leaves loss_eps=0.004 and pose_eps=0.0. pose_eps=0
    # makes the pose half of the AND criterion trivially true, so the rule is
    # loss-only, and 0.004 is loose enough that slow progress reads as
    # convergence.
    #
    # Measured, ES_MIN=25, cap 200: frame 325 - the frame EVERY run bumps at,
    # baseline included - stopped after 28 iterations. The distribution was
    # 332 frames under 40, 225 at 40-79, 25 at 80-149, 9 at 150+, so budget WAS
    # being spent, just not where the sequence is hard. Allocation that misses
    # the one section that needs it is not allocation.
    #
    # 1e-4/1e-4 is the reference pair from the base config's own comments.
    config["tracking"]["early_stop"]["loss_eps"] = float(
        os.environ.get("ES_LOSS_EPS", "1e-4"))
    config["tracking"]["early_stop"]["pose_eps"] = float(
        os.environ.get("ES_POSE_EPS", "1e-4"))

    config["run_name"] += f"_es{config['tracking']['early_stop']['min_iters']}"
    _le = os.environ.get("ES_LOSS_EPS", "1e-4")
    if _le != "1e-4":
        config["run_name"] += f"le{_le}"
    _rt = os.environ.get("ES_RETUNE", "0")
    if _rt != "0":
        config["run_name"] += f"rt{_rt}"
else:
    # ES=0 (or unset) WAS A NO-OP: splatam_es_graph.py sets early_stop.enabled
    # = True unconditionally in the inherited chain ("kept, already on in the
    # base"), and this whole block only ever ran when ES was truthy - so an
    # ES=0 caller (every "opts off" arm in this record, including
    # run_splatam_precond_suite.sh's ES=0) silently got early_stop ON anyway,
    # at whatever min_iters/loss_eps/pose_eps the chain inherited. This is the
    # actual off switch.
    config["tracking"]["early_stop"]["enabled"] = False

# THE MECHANISM PROBE. ATE at a fixed budget reports the outcome AFTER the
# estimator has had every iteration it wanted, so it cannot test the claim that
# the rank-1 estimator is slow to learn. i40, i30 and i20 all agreed between
# the arms, which is what a saturated budget looks like, not a refutation.
#
# PRE_PROBE=1 runs the rank-1 estimator as a SHADOW alongside the sample metric
# on the same gradient stream, and records cos(M_sample, M_rank1) against the
# within-frame iteration index. The catch-up iteration k is then a measured
# number: below k the sample metric has something to offer, above it nothing.
#
# NEVER QUOTE TIMING FROM A PROBE RUN. Each row costs a 6x6 eigh and several
# host syncs inside the tracking loop.
config["tracking"]["precond_probe"] = dict(
    enabled=os.environ.get("PRE_PROBE", "0") not in ("0", "", "false", "False"),
    every=int(os.environ.get("PRE_PROBE_EVERY", "20")),
    beta2=float(os.environ.get("PRE_BETA2", "0.95")),
    path=os.environ.get(
        "PRE_PROBE_PATH",
        f"../results/precond_probe_{config['run_name']}.jsonl"),
)

# Build the default trace path only after EVERY experimental tag has been
# appended. Constructing it beside the base run_name made max-step/restart
# arms collide in one file - the same silent overwrite class this config has
# already had to fix for trajectories and probes.
if _trace_req:
    _trace_path = (_trace_req if _trace_req != "1" else
                   f"../results/excursion_{config['run_name']}.jsonl")
    _excursion_trace = dict(
        path=_trace_path,
        history=int(os.environ.get("PRE_EXCURSION_HISTORY", "128")),
        post=int(os.environ.get("PRE_EXCURSION_POST", "24")),
        raw_step_ratio=float(os.environ.get("PRE_EXCURSION_RAW", "4.0")),
        seen_drop_ratio=float(os.environ.get("PRE_EXCURSION_SEEN", "0.20")),
        loss_spike_ratio=float(os.environ.get("PRE_EXCURSION_LOSS", "0")),
        trigger_frame=int(os.environ.get("PRE_EXCURSION_AT", "-1")),
        trigger_iteration=int(os.environ.get("PRE_EXCURSION_AT_ITER", "0")),
    )
else:
    _excursion_trace = {}

# Read by the tracking loop, not by PosePreconditioner: the observation is a
# function of the per-iteration loss alone, so it needs none of the class's
# internals and the class needs no new argument.
config["tracking"]["diag_shadow"] = dict(
    enabled=_diag_shadow,
    calibration_frames=_diag_calib,
    patience=_diag_patience,
    min_iter=_diag_min,
)

config["tracking"]["preconditioner"] = dict(
    enabled=_on,
    # THE STOPPING CRITERION THAT MATCHES THIS OPTIMISER. Stop when the
    # gradient norm has fallen to this fraction of the frame's OWN first
    # gradient. 0 disables and falls back to the early-stop machinery.
    #
    # Neither existing criterion works for a preconditioned tracker:
    #   pose_eps can never fire - M ~ E[gg^T] scales as g^2 and m as g, so the
    #     step is exactly scale-free and stays ~lr at the optimum forever
    #     (measured: pose_eps=1e-4 against a ~1e-3 step pinned every frame at
    #     the 200 cap);
    #   loss_eps has no transferable scale - 0.004 stopped the hard frame 325
    #     after 28 iterations, 1e-4 stopped nothing at all.
    #
    # |g|/|g_0| is scale-free ACROSS frames because each frame normalises by
    # its own start, so one threshold covers easy and hard frames alike. That
    # is the no-per-scene-tuning property the method is arguing for, applied to
    # its own stopping rule.
    # AUTO_LR=1 derives the step magnitude from this config's own Adam lrs
    # (sqrt(3 lr_trans^2 + 3 lr_rot^2)) and ignores PRE_LR. That removes the
    # one per-dataset constant the method still had - the metric supplies the
    # direction, which is the contribution, and Adam's already-tuned lrs
    # supply the scale, which never was.
    auto_lr=os.environ.get("AUTO_LR", "0") not in ("0", "", "false", "False"),
    stop_rel_grad=float(os.environ.get("STOP_REL", "0.0")),
    # STOP_MODE=best turns the gradient rule from a THRESHOLD into a PATIENCE.
    # STOP_REL then only acts as the on-switch (any value > 0 enables stopping);
    # STOP_PATIENCE and STOP_IMPROVE are the real knobs.
    #
    # Measured reasons the threshold form cannot work: |g|/|g_0| is NON-MONOTONE
    # (1.000 0.847 1.043 0.768 0.877 0.843 0.684 0.619 at it 0/1/2/5/10/20/30/40)
    # and plateaus near 0.6 because the step is scale-free and the optimiser
    # orbits rather than approaches. A loose threshold fires on a noise dip; a
    # strict one waits ~90 iterations for the plateau to drift under it.
    stop_mode=os.environ.get("STOP_MODE", "rel"),
    stop_patience=int(os.environ.get("STOP_PATIENCE", "10")),
    # FINAL_DENSE=1: run ONE full-resolution iteration at the stopping moment.
    #
    # is_sparse_phase masks 0.0 <= iter/(num_iters-1) < 1.0, so the ONLY dense
    # iteration in a frame is the last one - and a frame that stops early never
    # reaches it. Confirmed exactly: the ES+retune run stopped 288/591 frames
    # early and logged 591-288 = 303 unmasked iterations; the plateau run
    # stopped every frame early and logged 0 unmasked out of 14047, never
    # looking at a full image before committing a pose, at ATE 7.96 against
    # 3.0-3.5 for arms that did. Costs ~1 render in 24.
    final_dense=os.environ.get("FINAL_DENSE", "0") not in ("0", "", "false", "False"),
    stop_min_iters=int(os.environ.get("STOP_MIN", "10")),
    # STOP_REF: what |g| is judged against.
    #   frame   - this frame's own |g_0|. DIVERGENCE-UNSAFE: a wrong pose gives
    #             an enormous |g_0|, so a 20x drop is nearly free, the frame
    #             quits at the floor while still wrong, and the next frame
    #             starts worse. Divergence makes the criterion fire SOONER.
    #             Seen directly: every frame stopping at exactly STOP_MIN=10
    #             with mapping down to 2 it/s. Raising the floor does not fix
    #             it - that changes how fast the spiral runs, not whether.
    #   running - a slow average of per-frame INITIAL |g|, i.e. what a typical
    #             frame starts from. Healthy frames behave as before; a
    #             diverged frame must reach the level a normal frame reaches,
    #             cannot do it quickly, runs to the cap, and gets a chance to
    #             recover.
    # Default 'frame' so every measured run stays reproducible.
    stop_ref=os.environ.get("STOP_REF", "frame"),
    # float() on a device tensor is a HOST SYNC in the hot loop, and at
    # ~67 iterations per frame that is 67 of them to save at most a few
    # iterations. Checking every k costs up to k-1 extra iterations and
    # removes (k-1)/k of the syncs.
    stop_check_every=int(os.environ.get("STOP_EVERY", "5")),
    # STAGE 1: precondition on the SE(3) tangent. 6 parameters for 6 DOF, so
    # there is no gauge direction for the Levenberg floor to pour its longest
    # step into - the failure that took three patches to contain in Stage 0,
    # gone by construction rather than by projection. The render path is
    # unchanged; only the step moves to the tangent, via an analytic Jacobian
    # validated against central differences at cos=1.0000000000.
    # PRECOND_TANGENT=0 reproduces the 7-parameter Stage 0 behaviour.
    tangent=os.environ.get("PRECOND_TANGENT", "1") not in ("0", "", "false"),
    # PRE_LR IS NOT ADAM'S lr AND MUST BE SWEPT SEPARATELY. Adam's update is
    # bounded at ~lr per coordinate; the preconditioned one is lr * P m, and
    # P's eigenvalues are 1/sqrt(lambda + tau*lambda_max) - along a fresh M's
    # dominant direction that is already ~4.5x. Running it at Adam's 0.002
    # steps several times harder than Adam, which is exactly the failure the lr
    # sweep identified (rotation over-stepping wrecks tracking here).
    #
    # This is not a retreat from "the method removes the tuning". It replaces
    # SIX coupled quantities - the relative scaling between all pose DOF - with
    # ONE global step size. The shape is learned; the scale still has to be set,
    # exactly as it does for Adam.
    #
    # Read the "step |d| mean ... = N.NNx Adam@lr" figure in the end-of-run
    # summary and pick PRE_LR so that ratio lands near 1.
    lr=float(os.environ.get("PRE_LR", "0.002")),
    # PRE_BETA1=0 gives pure BFGS: the theory uses g, not an EMA of g.
    # The secant pair stays valid either way - s is whatever step was
    # applied - but momentum is one more thing between estimator and result.
    beta1=float(os.environ.get("PRE_BETA1", "0.9")),
    # NOT Adam's 0.999. That horizon is ~1000 steps, which inside a 40-90
    # iteration frame never adapts at all. The cross-frame carry supplies the
    # initialisation; beta2 supplies the within-frame refinement, and that
    # wants a horizon of tens of iterations.
    # PRE_BETA2 sets the metric's memory: horizon ~ 1/(1-beta2), so 0.95 is
    # ~20 iterations. That is the knob controlling HOW MUCH THE CARRY IS
    # WORTH. At 0.95 over a 40-iteration frame the carried M has decayed to
    # 0.95^40 = 0.13 by the end - the amortisation is discarded exactly where
    # the budget is tightest and it should matter most. A longer horizon
    # makes each frame lean on the carry instead of re-estimating from
    # scratch, which is the amortisation claim made operational.
    beta2=float(os.environ.get("PRE_BETA2", "0.95")),
    # Levenberg floor on the spectrum. Bounds the longest/shortest step ratio
    # at sqrt((1+tau)/tau) ~ 4.6, however singular M is - and M is singular by
    # construction for the first n steps of a cold frame, since gg^T is rank 1.
    # PRE_TAU: the Levenberg floor. It sets how much the metric is allowed
    # to reshape the step - the longest/shortest ratio is
    # sqrt((1+tau)/tau), so 4.6x at 0.05 and 7.1x at 0.02. LOWER tau means
    # MORE preconditioning, which is the mechanism under test, so it is
    # the knob to try when iterations will not come down.
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
    # PRE_SAMPLES: estimate M from the PER-GAUSSIAN gradient decomposition the
    # backward already computes, instead of one rank-1 g g^T per iteration.
    #
    # WHAT IT ATTACKS. The line above says M is 21 parameters fed by rank-1
    # updates. This is the other way to answer that: not shrink the matrix
    # toward a diagonal it can estimate, but feed it enough samples to estimate
    # the whole thing. In isotropic mode the pose reaches the loss through
    # transformed_pts alone, so its retained grad is a [N,6] sample matrix G
    # with sum(G) == the gradient already being used - the same total, grouped
    # instead of summed. Full rank from the first backward; the rank-1 EMA
    # needs six iterations before M can even be invertible.
    #
    # AND IT IS THE EXPERIMENT THAT RESCUES THE FLAT GAMMA SWEEP. Flat over
    # 1.0 -> 0.5 is consistent with TWO stories: the coupling does not matter,
    # or OUR ESTIMATE of the coupling is noise, in which case shrinking noise
    # toward a diagonal would also change nothing. A well-estimated M
    # separates them. Re-run the gamma sweep with PRE_SAMPLES=1: still flat
    # means story one and the chapter is about the tangent plus the stopping
    # rule; not flat means the full-matrix claim is real and was invisible
    # because the estimator could not see it.
    #
    # Cannot run with either CUDA graph - the reduction is not fixed-shape.
    # splatam.py refuses rather than replaying a frozen metric.
    sample_metric=_samples,
    stop_improve=float(os.environ.get("STOP_IMPROVE", "0.01")),
    # PRE_HANDOFF=N: the preconditioner runs the first N iterations of each
    # frame, then Adam finishes it.
    #
    # WHY IT MIGHT WORK. M^-1/2 m is exactly scale-free - the step stays ~lr at
    # the optimum forever, which is why pose_eps can never fire. So the
    # preconditioner has a structural reason to be good at ACQUIRING a basin
    # and indifferent at REFINING inside one, and more iterations cannot fix
    # that. The measured curve is consistent: i20 -> 3.9 cm, i40 -> 3.5 cm, so
    # doubling the budget buys 0.4 cm.
    #
    # WHY IT MIGHT NOT. Adam is 0/10 in this config at every budget from 10 to
    # 40 iterations, so "Adam refines well" is unevidenced - though that is
    # measured from a PREDICTED pose, and refining from a good one is a
    # different task. Also the headroom is small: main branch at full budget is
    # 3.22 cm, so ~3.2 looks like the MAP's floor, not the tracker's.
    #
    # COMPARE AT EQUAL TOTAL RENDERS. handoff=10 with ITERS=20 against ITERS=20
    # with no handoff. Comparing against a longer run measures the budget.
    handoff=int(os.environ.get("PRE_HANDOFF", "0")),
    # PRE_M0=iso: seed M at each cold frame start with (|g_0|^2/n) I instead of
    # zero. ONLY meaningful with CARRY=0 - with the carry there is nothing cold
    # to seed beyond frame 1.
    #
    # WHY THE CARRY-OFF COMPARISON NEEDS IT. From M=0 the two arms differ in
    # RANK, not just shape: the rank-1 arm is singular for its first 6 steps and
    # leans on the Levenberg floor, while the sample arm is full rank from its
    # first backward. Rank is not the variable under test. The isotropic seed
    # gives both arms identical initial scale (trace = |g_0|^2, the same
    # convention trace matching uses) and identical rank, leaving SHAPE as the
    # only difference. If the sample metric is still null under that, Level 2 is
    # closed on TUM.
    m0_iso=os.environ.get("PRE_M0", "") == "iso",
    # BFGS=1: secant mode. M = E[gg^T] answers "how are my gradients
    # distributed" - a gradient covariance. Level 2 showed that estimating it
    # better buys nothing, so this changes the KIND of information instead:
    # y_k = g_{k+1} - g_k against s_k = the step actually applied, which for a
    # locally quadratic objective satisfies y = H s exactly. Both are free -
    # the step is known and the next backward is already paid for.
    #
    # RUN IT WITH AUTO_LR=0 AND CARRY=0. NOT AUTO_LR=1 - that was the first
    # recommendation and it was wrong. target_step_norm fixes every step at
    # one length, which is harmless for M (already scale-free) and FATAL for
    # BFGS: -B g with B ~ H^-1 must SHRINK as the gradient shrinks, and pinning
    # it removes the only property BFGS provides. Measured as mean |d| ==
    # max |d| == 1.00x Adam over 2490 steps, ATE 125 cm. splatam.py now
    # refuses the combination. PRE_LR sets the FIRST step of each frame and
    # the secant equation carries the scale from there.
    #
    # CARRY=0 because the carry, the transport, the EMA and the initialisation
    # are already four entangled knobs - let B learn from the current frame's
    # pairs alone before adding a fifth.
    #
    # PRE_BETA1=0 gives the pure form: BFGS theory uses g, not an EMA of g.
    # The secant pair stays valid either way (s is whatever step was applied),
    # but momentum is one more thing between the estimator and the result.
    bfgs=os.environ.get("BFGS", "0") not in ("0", "", "false", "False"),
    bfgs_max_cond=float(os.environ.get("BFGS_COND", "20.0")),
    # BFGS_CURV is an ANGLE threshold: y^T s / (|y| |s|). 1e-8 was the first
    # value and is not a threshold at all - it accepts near-flat pairs where
    # rho = 1/(y^T s) is enormous, and ONE such pair took |B| from 2.45 to 1e8
    # in a unit test. Measured consequence in a real run: B grew ~10x inside a
    # frame and max |d| sat exactly on the trust-region cap.
    bfgs_curv_eps=float(os.environ.get("BFGS_CURV", "1e-2")),
    # How far B may grow within a frame relative to its autoscaled start.
    # clamp_spectrum bounds cond(B) but NOT its magnitude, so this is a
    # separate guard, not a tighter max_cond.
    bfgs_max_growth=float(os.environ.get("BFGS_GROWTH", "4.0")),
    # B_0 from this config's own tuned Adam lrs rather than the identity -
    # rho and theta do not share a scale, and the measured lr sweep (opposite
    # -signed dATE/dlr for rot and trans) is exactly that statement.
    bfgs_b0_from_lrs=os.environ.get("BFGS_B0", "lrs") == "lrs",
    # eigh is not capturable and costs a solve; the metric changes slowly
    # enough that rebuilding it every 10 steps is ample. If tracking
    # ms/iteration rises noticeably, this is the first knob to turn.
    refactor_every=10,
    # Trust region, as a multiple of Adam's lr*sqrt(n). The spectrum damping
    # bounds P's CONDITION NUMBER; nothing else bounds its SCALE, and one
    # oversized step is unrecoverable here because the map is then built from
    # the wrong pose. Without this the first run measured a step norm 4258x
    # Adam's and a quaternion norm of 2915 by frame 5.
    # PRE_MAX_STEP makes this sweepable, because 10 is a GUESS and it is a
    # loose one. Adam's per-coordinate step is ~lr by construction: m and v
    # come from the same gradient stream, so m_hat/sqrt(v_hat) is ~1 when
    # gradients are consistent and <1 when they are noisy. That is a bound on
    # EVERY coordinate. This cap bounds only the total norm, at 10x Adam's
    # scale, and nothing stops the metric pouring the whole budget into one
    # weakly-observed direction.
    #
    # Both divergent handoff-free runs reported step |d| max 2.45e-01 -
    # exactly 10 * 0.01 * sqrt(6), the cap, to three digits. In those runs the
    # TRUST REGION was the only thing bounding the step, and it was set an
    # order of magnitude looser than the optimiser it is being compared to.
    max_step_mult=float(os.environ.get("PRE_MAX_STEP", "10.0")),
    # C1. PRE_RESTART=N: zero the first moment at within-frame iteration N,
    # keeping M and P. THE HANDOFF ALREADY DOES THIS - splatam.py asserts
    # Adam's state is empty at the switch, and prints a warning if it is not -
    # so the 6/6 arm is a change of update rule AND a momentum restart, and
    # nothing on record separates them. This is the restart alone.
    #
    # If it recovers most of the handoff's stability, the second CUDA-graph
    # capture per frame was never buying anything and the method runs a whole
    # frame by itself. Refused with PRE_HANDOFF>0: it is driven from
    # refactor(), which is not called during the Adam phase, so the
    # combination would silently do nothing at all.
    restart_at=_restart,
    # PRE_RESTART_M: what the restart does to M. "off" clears the first moment
    # only. "trace" also re-anchors M's SCALE to the current gradient while
    # keeping its SHAPE - the coupling the carry exists to accumulate is a
    # scalar multiply away and survives untouched, and only the magnitude,
    # which is stale by construction while |g| falls, is corrected. "iso"
    # discards the shape too, as the contrast that can falsify the coupling
    # claim at that point in the frame.
    #
    # MEASURED, TUM fr1, it20-39, as multiples of ref: the metric steps at
    # 0.10x where the Adam tail steps at 0.33x. Clearing m alone gives 0.19x,
    # matching the EMA horizons (PRE_BETA2=0.9) 0.14x, both together 0.21x -
    # so half the gap is M's scale, which neither touches. On a decaying
    # gradient trace(M) ends a frame 34x above |g|^2, and P ~ M^-1/2 makes that
    # a ~5.9x step suppression exactly when the frame has stopped converging.
    restart_m=_restart_m,
    # PRE_PROFILE_EVERY=N: print the step profile every N frames and reset it,
    # so each report is a WINDOW. Needed for any full-length diagnostic - a
    # cumulative profile averages the frames where the tracker breaks into the
    # hundreds where it does not. Pure instrumentation, so it carries no
    # run_name tag: it reads accumulators and zeroes them and cannot change a
    # trajectory. 0 = off.
    profile_every=int(os.environ.get("PRE_PROFILE_EVERY", "0")),
    # LR_SERIES=1: print the raw per-frame IT/FRAME, REL-GRAD, K series in
    # the summary. Off by default - lr_ladder.sh sets it for the runs it
    # hands to profiling/lr_proxy_replay.py; nobody else needs it.
    log_series=os.environ.get("LR_SERIES", "0") not in ("0", "", "false", "False"),
    excursion_trace=_excursion_trace,
    # DRIFT_MULT: bar a frame from stopping early when its mean pre-cap step
    # |d|/ref exceeds this multiple of a baseline frozen after DRIFT_WARMUP
    # frames. 0 = off.
    #
    # THE FAILURE IT IS FOR, measured on a diverging full-length run:
    #   window      it5-9 mean   barred
    #   250-275       0.20x       11%
    #   275-300       0.33x       10%
    #   300-325       0.35x       10%
    #   325-350       0.84x        9%   <- frustum exit begins here
    # The existing anomaly guard never moved while the step distribution
    # quadrupled, because its reference is an EMA over a ~50-frame horizon and
    # the degradation takes ~40 - so it tracked the decay instead of flagging
    # it. This guard normalises by ref = lr*sqrt(n), a constant, and freezes
    # its baseline, so neither half can drift.
    #
    # ONLY THE MULTIPLIER IS A CONSTANT, and it is dimensionless - the level it
    # is compared against is learned from the scene's own healthy frames.
    # PRE_DEAD_FREEZE=0 restores the OLD behaviour, for an A/B only.
    #
    # ON BY DEFAULT because it is a BIT-EXACT no-op on any run without empty
    # renders (test 32 asserts identical M, m and every step over 40
    # iterations) - it can only alter runs that were already broken.
    #
    # WHAT IT FIXES. |g| is exactly zero when the pose leaves the frustum, but
    # the EMA still ran: M *= beta2 per iteration with nothing added, so M
    # collapsed geometrically and P = M^{-1/2} exploded. Measured on a
    # diverging run with 923 empty iterations: a requested step of 9859x ref.
    # That is the positive feedback which makes a brief excursion permanent -
    # the first gradient that comes back is multiplied by a metric orders of
    # magnitude too large.
    #
    # AND IT EXPLAINS A STANDING PUZZLE. The record notes that divergent runs
    # report "max |d| = 2.45e-01 EXACTLY - the trust-region cap... a value
    # repeating to three digits across configs is a constant, not data.
    # Symptom or cause is not yet known." This is the cause: with P exploded
    # every step saturates the cap, so max |d| reads the cap whatever the
    # config. Test 32 reproduces it at 10.0x on demand.
    dead_freeze=os.environ.get("PRE_DEAD_FREEZE", "1") not in ("0", "", "false", "False"),
    drift_mult=float(os.environ.get("DRIFT_MULT", "0")),
    drift_warmup=int(os.environ.get("DRIFT_WARMUP", "50")),
    # C2. PRE_RAMP="from:to" ramps the applied step from the metric's toward
    # normalised gradient descent at Adam's magnitude. "N" is a hard switch at
    # N - the handoff's exact shape, with one optimiser, one state and ONE
    # CAPTURE PER FRAME, which is the whole reason removing the handoff was
    # ever worth anything.
    #
    # NOT THE SAME AS PRE_SHRINK=0, and this is why it is a separate knob:
    # shrink pulls M toward diag(M), Adam's LEARNED per-coordinate scales in
    # the tangent. This pulls the step toward the identity DIRECTION, with no
    # learned scale at all. The gamma sweep does not cover it.
    ramp_from=_ramp_from,
    ramp_to=_ramp_to,
    # PRE_DIAG_AFTER=N: full M^{-1/2} for acquisition, then
    # diag(M)^{-1/2} for refinement. m and M are continuous across the switch;
    # only the applied factor changes, and each new frame starts full again.
    diag_after=_diag_after,
    adaptive_diag=_adaptive_diag,
    adaptive_diag_patience=_diag_patience,
    # THE CALIBRATION SETTINGS BELONG TO THE APPLYING PATH ONLY. In shadow
    # mode the class runs an ordinary fixed diag_after and knows nothing about
    # the measurement - the tracker keeps the samples and installs the median -
    # so these must be zero or the constructor refuses them:
    #   calibration_frames > 0 requires adaptive_diag=True
    #   min_iter > 0 requires calibration_frames > 0
    # Both fire at construction, which is where the first two shadow launches
    # died. The shadow settings travel in tracking.diag_shadow instead.
    adaptive_diag_calibration_frames=(_diag_calib if _adaptive_diag else 0),
    adaptive_diag_min_iter=(_diag_min if _adaptive_diag else 0),
    # PER-COORDINATE RATES FOR THE TAIL ONLY - the full-matrix phase keeps
    # PRE_LR. None preserves the shared-rate behaviour every arm in the ladder
    # was measured with. The class refuses a non-None value without a tail;
    # the guard at parse time catches that first and says which flag did it.
    diag_lr=_diag_lr,
    # STOP_ANCHOR=sig uses the standard early-stopping definition, where the
    # plateau anchor moves only on an improvement that beat the margin. The
    # default "min" is the RECORDED behaviour - a ratchet that slides the
    # anchor to every new minimum and therefore cuts frames that are still
    # converging. Default left unchanged so every number already measured
    # stays comparable; see test group 25.
    stop_anchor=os.environ.get("STOP_ANCHOR", "min"),
    # The whole point: M is a property of the scene geometry seen from the
    # camera and adjacent frames share it, so the next frame starts with a
    # full-rank metric instead of spending its first n iterations estimating
    # one. Set False to measure exactly what the carry is worth.
    carry=os.environ.get("CARRY", "1") not in ("0", "", "false", "False"),
    # TEST OF THE OTHER HALF OF THE CARRY CLAIM. Default False reproduces
    # every existing run exactly (see PosePreconditioner's own constructor
    # comment). Set CARRY_M=1 to also carry m across the frame boundary
    # instead of resetting it - the file's stated reason not to is that the
    # constant-velocity re-init makes the previous frame's last gradient
    # direction stale; this is how that claim gets checked instead of assumed.
    carry_m=os.environ.get("CARRY_M", "0") not in ("0", "", "false", "False"),
    # ADJOINT TRANSPORT of that carry. M lives in the tangent at the current
    # pose, which for a left perturbation is in CAMERA-frame coordinates - and
    # the camera moves between frames, so M_{t-1} and M_t are stated in
    # different coordinate systems. Copying mixes them; A^-T M A^-1 with
    # A = Ad_{T_t T_{t-1}^-1} is the correction.
    #
    # DEFAULT OFF, deliberately: every result measured so far ran without it,
    # and flipping the default would silently make them unreproducible. The
    # untransported carry is first-order accurate (adjacent frames give A ~ I),
    # which is why it works at all - transport makes it exact, and the error it
    # removes is first-order in inter-frame motion: small per frame,
    # systematic, and accumulating over 592 of them.
    #
    # Only meaningful in the tangent. Stage 0's quaternion coordinates have no
    # adjoint, which is why this could not be tested until Stage 1 landed.
    transport=os.environ.get("TRANSPORT", "0") not in ("0", "", "false", "False"),

    # THE ANOMALY GUARD, now measurable instead of hard-coded.
    #
    # A frame whose starting |g| exceeds ANOM_MULT x the running reference is
    # barred from early stopping and runs the FULL budget. That makes these
    # three numbers a large share of tracking cost, and until now none of them
    # was settable or reported.
    #
    # THE FAILURE THEY EXIST TO EXPOSE. Only an ACCEPTED frame updates the
    # reference, so the reference cannot learn a level it does not already
    # accept. profiling/anom_ratchet.py reproduces the consequence on CPU in
    # seconds: a transient spike recovers and a gradual ramp is tracked, but a
    # SUSTAINED 3x shift in gradient level froze the reference permanently and
    # barred 99 of the following 100 frames. fr1_desk is known to change
    # regime around frame ~325 ("every run bumps at frame 325"), and ~46% of
    # frames run to the cap on the 15x arm - consistent, but NOT yet confirmed
    # to be the same thing. The summary now prints "barred N/M", which is what
    # confirms or kills it.
    #
    # ANOM_MULT   how far above the reference still counts as healthy (2.0).
    # REF_DECAY   0.98 is a ~50-frame horizon. The carry argues that adjacent
    #             frames share scene structure; a 50-frame average throws that
    #             locality away. 0.8 is a ~5-frame horizon.
    # BARRED_ADMIT  rate at which a BARRED frame may still move the reference.
    #             0.0 is the shipped behaviour and is what makes the freeze
    #             absorbing. A small value absorbs a persistent new level over
    #             ~1/rate frames while still lagging a divergence spiral, whose
    #             |g_0| grows every frame rather than settling at a plateau.
    anom_mult=float(os.environ.get("ANOM_MULT", "2.0")),
    ref_decay=float(os.environ.get("REF_DECAY", "0.98")),
    barred_admit=float(os.environ.get("BARRED_ADMIT", "0.0")),
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
# GRADSTALE=1 runs the gradient-staleness probe INSIDE this config, so it
# can be driven through the suite and therefore measured at the SHIPPED
# operating point. That matters more than it sounds: run standalone this
# config takes 200 iterations a frame with WANDER around 7, while the
# suite's PRESET=final leaves ~27 iterations and WANDER around 3.2. The
# probe asks how fast a gradient goes stale, and it goes stale FASTER in a
# more oscillatory regime - so the standalone number is the pessimistic
# end of the range, not the operating point.
#
# TAGGED, and graphs forced off: the probe runs extra forward+backward
# passes inside the tracking loop, which launch into a non-capturing
# stream under an active capture.
if os.environ.get("GRADSTALE", "0") not in ("0", "", "false", "False"):
    # GRADSTALE_DENSE=1: nearly every consecutive-iteration pair instead of
    # the 6-point go/no-go (starts=[5,15], offsets=[1,2,4]) - the shipped-
    # operating-point counterpart of SplaTAM/configs/tum/
    # splatam_gradstale_adam_dense.py's Adam-arm trace, for a per-frame
    # trend plot (profiling/plot_grad_staleness_dense.py) rather than a
    # table.
    #
    # RANGE COVERS THE FULL 200-ITERATION CAP (iters=200 for fr1_desk in
    # run_splatam_precond_suite.sh), NOT just the ~25-27 iterations the
    # shipped WCONV/STOP_REL stopping actually leaves. GradStalenessProbe's
    # wants() is a no-op past whatever iteration the frame actually stops
    # at, so this same range is correct whether stopping is left on (it
    # naturally truncates, same as the sparse probe always has) or disabled
    # with the suite's own NOSTOP=1 (WCONV=0, STOP_REL=0) for full coverage -
    # one flag serves both cases, no separate "_full" variant needed the way
    # the Adam arm's early_stop required.
    #
    # STARTS ARE ODD FOR THE SAME REASON AS THE ADAM VARIANT:
    # GradStalenessProbe.wants() gives `iter_idx in starts` priority over
    # closing the active window, so back-to-back integer starts with
    # offsets=[1] silently drop most measurements - see
    # splatam_gradstale_adam_dense.py's header for the full mechanism.
    if os.environ.get("GRADSTALE_DENSE", "0") not in ("0", "", "false", "False"):
        _gs_starts, _gs_offsets, _gs_out = (
            list(range(1, 200, 2)), [1], "grad_staleness_dense.jsonl")
    else:
        _gs_starts, _gs_offsets, _gs_out = (
            [5, 15], [1, 2, 4], "grad_staleness.jsonl")
    config["tracking"]["grad_staleness"] = {
        "enabled": True,
        "frames": [50, 150, 250],
        # Inside the ~25-27 iterations PRESET=final actually leaves, so
        # the deepest offset (15 + 4) still lands in a real frame.
        "starts": _gs_starts,
        "offsets": _gs_offsets,
        "usable_cos": 0.95,
        "out_path": _gs_out,
    }
    config["tracking"]["cuda_graph"]["enabled"] = False
    config["tracking"].setdefault("iteration_graph", {})["enabled"] = False
    config["run_name"] += "_gstale"

# ONLINE ACQUISITION-LR TUNER, TAGGED INTO run_name.  The implementation is
# shared, but SplaTAM uses the scalar preconditioner lr (AUTO_LR=0) rather than
# MonoGS's target_step_norm.  Without this tag an online arm would overwrite
# the fixed-PRE_LR artifact it exists to compare against.
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

# GRADIENT REUSE, TAGGED INTO run_name.
#
# The flags themselves are read in splatam.py (utils/grad_reuse.py's
# config_from_env), so run_name carried NO trace of them - every reuse arm
# so far was separated from its baseline only by a REP the operator had to
# remember to set. run_name keys the output directory and params.npz, so
# one forgotten REP silently overwrites the arm it exists to be compared
# against. That is the bug class this repo has hit nineteen times.
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
        # LOCAL_TRUST/TRUST_MAG were unconditional on GRAD_REUSE_ADAPTIVE
        # but had NO tag at all here - a local_trust run (drops the boundary
        # ratchet entirely) and a ratchet run of otherwise identical
        # settings would write to the same output directory/params.npz.
        if os.environ.get("GRAD_REUSE_LOCAL_TRUST", "0") not in (
                "0", "", "false", "False"):
            config["run_name"] += "lt"
        if os.environ.get("GRAD_REUSE_TRUST_MAG", "0") not in (
                "0", "", "false", "False"):
            _gr_maglo = os.environ.get("GRAD_REUSE_TRUST_MAG_LOW", "0.8")
            _gr_maghi = os.environ.get("GRAD_REUSE_TRUST_MAG_HIGH", "1.2")
            config["run_name"] += (
                "mg%s_%s" % (_gr_maglo.replace(".", ""),
                             _gr_maghi.replace(".", ""))
            )
        _gr_reject_patience = os.environ.get("GRAD_REUSE_REJECT_PATIENCE")
        if _gr_reject_patience is not None:
            config["run_name"] += "rp%s" % _gr_reject_patience
        _gr_reject_backoff = os.environ.get("GRAD_REUSE_REJECT_BACKOFF")
        if _gr_reject_backoff is not None:
            config["run_name"] += "rb%s" % _gr_reject_backoff
        _gr_stop_clock = os.environ.get("GRAD_REUSE_STOP_CLOCK", "")
        if _gr_stop_clock == "render":
            config["run_name"] += "rc"
        elif _gr_stop_clock == "fresh":
            config["run_name"] += "fc"
        if os.environ.get("GRAD_REUSE_ASYNC_CHECK", "0") not in (
                "0", "", "false", "False"):
            config["run_name"] += "ac"
        if os.environ.get("GRAD_REUSE_DEFER_CHECK", "0") not in (
                "0", "", "false", "False"):
            config["run_name"] += "dc"
        if os.environ.get("GRAD_REUSE_BATCHED_CHECK", "0") not in (
                "0", "", "false", "False"):
            config["run_name"] += "bc%s" % os.environ.get(
                "GRAD_REUSE_TRUST_LEASE", "2")

_pixel_sample_env = os.environ.get("PIXEL_SAMPLE")
if _pixel_sample_env is not None:
    _ps_on = _pixel_sample_env not in ("0", "", "false", "False")
    config["tracking"]["pixel_sample"]["enabled"] = _ps_on
    # TAGGED, because run_name keys the output directory and params.npz. An
    # untagged sparse-off arm would overwrite the sparse-on run it exists to be
    # compared against - the bug class this repo has hit nineteen times.
    config["run_name"] += "_nops" if not _ps_on else "_ps"

# THE SPARSE OPERATING POINT. Same variable names MonoGS and GSLAM already
# read (slam_frontend.py, tracker.py), so one sweep script can drive all
# three rather than each model having its own spelling.
#
# WHY THIS IS WORTH SWEEPING AT ALL. The tile-cost probe measured the
# render stage at 31% of a tracking iteration, and cost share tracking tile
# share to within 0.033 - so render cost falls roughly LINEARLY with tiles
# dropped. At the shipped sample_ratio of 0.75 only a quarter of tiles go,
# capping the gain at ~7.8%, which is about what every sparse measurement
# on this branch has found. The mechanism was never broken; the operating
# point was timid. 0.25 has a ceiling of ~23%.
#
# AND gradient_frac IS NOT JUST A KNOB. The gradient-variance probe found
# gradient_frac=1.0 to be the only setting where no draw opposed the true
# pose gradient (worst-draw cos +0.22, against -0.48 at the shipped 0.5),
# with more than double the SNR. It is also deterministic, which removes
# the randperm redraw that gave GSLAM a 2 cm ATE swing on identical config.
# Its cost is a +20-25% gradient magnitude bias, which an optimiser sees as
# an effective learning-rate change - expect to retune PRE_LR with it.
#
# TAGGED, for the usual reason: run_name keys the output directory and
# params.npz, and the suite log name does NOT carry these, so two cells of
# a sweep would otherwise overwrite each other.
for _ps_key, _ps_env, _ps_tag in (("sample_ratio", "PS_RATIO", "r"),
                                  ("gradient_frac", "PS_GRADFRAC", "gf")):
    if _ps_env in os.environ:
        _ps_val = float(os.environ[_ps_env])
        config["tracking"]["pixel_sample"][_ps_key] = _ps_val
        config["run_name"] += "_%s%g" % (_ps_tag, _ps_val)
if os.environ.get("BINCAP", "1") in ("0", "", "false", "False"):
    # TAGGED: run_name keys the output directory, and a binning-off arm
    # must not overwrite the binning-on run it exists to be compared with.
    config["run_name"] += "_nobin"
