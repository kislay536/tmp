import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from diff_gaussian_rasterization import build_tracking_optimizer
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera, set_capture_safe_pose
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.early_stop import EarlyStop
from utils.pixel_sample import build_tile_mask, is_sparse_phase
from utils.grad_reuse import (GradReuse, RenderClock, stash_grads, restore_grads,
                              config_from_env as grad_reuse_env)
from utils.coarse_to_fine import downsample_image, downsample_depth, get_levels, get_iter_scale
from utils.binning_capacity import BinningCapacity
from utils.es_signals import ESSignalBuffer
from utils.tracking_iteration_graph import TrackingIterationGraph
from utils.iter_trace import IterTrace, rot_to_quat
from utils.pose_preconditioner import (
    PosePreconditioner,
    adaptive_failure_streak,
    adaptive_loss_improved,
)
from utils.windowed_convergence import (
    IncumbentConvergence,
    WindowedConvergenceSweep,
    make_windowed_convergence,
)


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        # Set by slam.py only when use_inline_backend is enabled - lets
        # this loop call self.backend.idle_mapping() directly (see run()
        # below), since the continuous background-mapping loop inside
        # BackEnd.run() never executes in that mode (nothing ever drives
        # it). Ported from RTGS (github.com/UMN-ZhaoLab/RTGS,
        # MonoGS_fullend branch).
        self.inline_backend = False
        self.backend = None
        self._idle_frame_counter = 0

    def _cam_lr(self, key):
        """The frontend's tracking learning rate for `key` (cam_rot_delta / cam_trans_delta).

        CAM_LR_SCALE multiplies BOTH rates. It exists because the Adam rates were only settable by
        editing a config, and MonoGS's ScanNet setup is a straight port of TUM's (upstream MonoGS
        has no ScanNet), while SplaTAM's stock ScanNet lr is 4x below its TUM value. It scales the
        FRONTEND's copy only - the backend (mapping) reads the config untouched - and every read
        of these two rates goes through here, so the Adam optimiser, the preconditioner's derived
        step norm (auto_lr) and the windowed stopper's normalisation stay consistent with each
        other. Unset or 1 returns the config value unchanged.
        """
        return float(self.config["Training"]["lr"][key]) * self._cam_lr_scale

    def set_hyperparams(self):
        _s = os.environ.get("CAM_LR_SCALE", "")
        self._cam_lr_scale = float(_s) if _s.strip() else 1.0
        if self._cam_lr_scale <= 0.0:
            raise ValueError("CAM_LR_SCALE must be > 0 (got %r)" % _s)
        if self._cam_lr_scale != 1.0:
            Log("[Tracking] CAM_LR_SCALE=%g: cam_rot_delta %g -> %g, cam_trans_delta %g -> %g" % (
                self._cam_lr_scale,
                float(self.config["Training"]["lr"]["cam_rot_delta"]), self._cam_lr("cam_rot_delta"),
                float(self.config["Training"]["lr"]["cam_trans_delta"]), self._cam_lr("cam_trans_delta")))
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.use_gui = self.config["Results"]["use_gui"]
        # TRACK_ITERS overrides the per-frame iteration cap without a new config.
        self.tracking_itr_num = int(os.environ.get(
            "TRACK_ITERS", self.config["Training"]["tracking_itr_num"]))
        # DISABLE_NATIVE_CONV=1: force MonoGS's own tau.norm() < 1e-4
        # convergence check to never fire, so every frame runs the full
        # tracking_itr_num budget - the native-stop equivalent of WCONV's
        # NOSTOP. This is a SEPARATE mechanism from Training.early_stop
        # (self.stopper below): early_stop.enabled=False only turns off the
        # shared loss/window criterion, native convergence is unconditional
        # and was not previously reachable by any env var (update_pose() is
        # always called with its default converged_threshold=1e-4 - checked
        # directly, all three call sites pass no threshold argument).
        self.native_conv_threshold = (
            -1.0 if os.environ.get("DISABLE_NATIVE_CONV", "0")
            not in ("0", "", "false", "False") else 1e-4
        )
        # SIGNAL_TRACE=<file>: one CSV row per tracking iteration with the values
        # the stopper reads (loss, step norm, converged flag, energy ratio,
        # proposal) and whether the iteration was a gradient-reuse one.
        # Diagnostics only; truncated at start so a rerun does not append.
        self._signal_trace_path = os.environ.get("SIGNAL_TRACE") or None
        if self._signal_trace_path:
            open(self._signal_trace_path, "w").close()
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        self.idle_mapping_interval = self.config["Training"].get(
            "idle_mapping_interval", 1
        )
        _es_cfg = dict(self.config.get("Training", {}).get("early_stop", {}))
        # ES=0/1 without editing the YAML, same name SplaTAM uses.
        #
        # WHY IT MATTERS FOR THE PRECONDITIONER ARM. The ladder config has
        # early_stop ON, so its loss criterion and the plateau criterion are
        # both armed and whichever fires first wins. On MonoGS the loss rule is
        # APPROPRIATE - the retuner measures tail_mass = 0.21 here against 1.00
        # on SplaTAM, i.e. only 21% of pose movement happens after the loss
        # stops improving - so it fires early and the plateau rule may never
        # get to act. Turning it off is what gives the new criterion room.
        if "ES" in os.environ:
            _es_cfg["enabled"] = os.environ["ES"] not in ("0", "", "false", "False")
        self.stopper = EarlyStop(_es_cfg)
        # Diagnostic: is the per-iteration .item() sync (needed to feed the
        # EMA'd pose delta into early-stop) eating the wall-time savings
        # that fewer iterations should buy? .item() forces a CPU-GPU sync,
        # which can stall the async kernel-launch pipelining between
        # iterations. Printed as a fraction of total tracking wall time at
        # end of run - remove once the early-stop wall-time gap is explained.
        self._sync_time_total = 0.0
        self._tracking_time_total = 0.0
        self._tracking_iters_total = 0
        self._mlsys_track_time = 0.0
        self._mlsys_track_frames = 0
        self._mlsys_track_iters = 0
        self._tracking_stop_counts = {
            "native": 0,
            "loss": 0,
            "plateau": 0,
            "window": 0,
            "cap": 0,
        }
        # TAIL DIAGNOSTIC. Answers "what do the late iterations buy?" without
        # running another arm. Per frame it records where the minimum loss
        # occurred, how much the objective fell after a reference iteration,
        # and how much worse the COMMITTED (final) pose is than the best one
        # visited - MonoGS commits the last pose, unlike SplaTAM which
        # restores the best-loss pose, so that gap is pure avoidable loss.
        self._tail_diag = os.environ.get("TAIL_DIAG", "0") not in (
            "0", "", "false", "False")
        self._tail_ref = int(os.environ.get("TAIL_DIAG_REF", "20"))
        self._tail_rows = []
        # Diagnostic: gating the GUI packet (use_gui=False) didn't move the
        # non-tracking per-frame wall time at all, so clone_obj wasn't the
        # (sole) dominant cost there. Break the rest of the per-frame body
        # down by section instead of guessing again.
        self._load_time_total = 0.0
        self._gui_time_total = 0.0
        self._kf_logic_time_total = 0.0
        self._eval_time_total = 0.0
        self._throttle_sleep_time_total = 0.0
        self._frame_loop_time_total = 0.0
        self._frame_loop_count = 0
        # 614.6s total - 540.2s tracking - 16.1s load - 17.4s kf_logic -
        # 3.9s eval - 1.5s throttle left ~35.5s unaccounted for. That gap
        # is the `else` branch of the main loop - receiving/applying
        # messages *from* the backend (sync_backend/keyframe/init), a
        # completely separate code path none of the above timers touch.
        self._backend_msg_time_total = 0.0
        self._backend_msg_counts = {}
        self._empty_cache_time_total = 0.0
        self._empty_cache_calls = 0
        # 629.6s total - 554.6s tracking - 16.2s load - 17.7s kf_logic -
        # 3.8s eval - 1.6s throttle - 4.7s backend_msg - ~0s empty_cache
        # still left ~31s unaccounted for, and none of those sections run
        # before the backend confirms initial map setup - these three
        # sleep(0.01)-and-continue gates at the very top of the loop (most
        # plausibly requested_init, waiting on the backend's initial
        # init_itr_num=1050-iteration map optimization) are the last
        # un-instrumented code the frontend can spend real time in.
        self._init_wait_time_total = 0.0
        self._single_thread_wait_time_total = 0.0
        self._not_initialized_wait_time_total = 0.0
        # pixel_sample diagnostic: nothing else distinguishes an enabled
        # run from a disabled one in the logs (unlike early-stop/adaptive-
        # mapping/render-cap, which all print their own engagement stats),
        # so a no-op-check run and a real sparse run look identical without
        # this.
        self._sparse_iters_total = 0
        self._corrective_render_count = 0
        self._ctf_iters_total = 0

        _train = self.config.get("Training", {})
        # A fixed binning capacity is the PREREQUISITE for the iteration graph,
        # not an independent optimisation: without it the rasterizer reads
        # num_rendered back to the host to size its binning buffer, and that one
        # synchronous memcpy is what makes it unrecordable.
        self.binning_capacity = BinningCapacity(_train.get("binning_capacity", {}))
        _ig_cfg = dict(_train.get("iteration_graph", {}))
        # ITERGRAPH=0 turns the capture off without editing the YAML. The
        # ladder config (fr1_desk_itergraph.yaml) has it ON, and the
        # preconditioner REFUSES to run with it - so testing the preconditioner
        # on top of the ladder needs a way to drop just this one piece.
        if "ITERGRAPH" in os.environ:
            _ig_cfg["enabled"] = os.environ["ITERGRAPH"] not in (
                "0", "", "false", "False")
        # ITERGRAPH_ERR / ITERGRAPH_WARMUP exist for the same reason ITERGRAPH
        # does: the Replica chain carries NO iteration_graph block at all, so
        # turning the graph on there falls back to capture_error_mode "global"
        # - which errors on any legacy-default-stream operation ANYWHERE in the
        # process during capture, including the backend's, and MonoGS/Replica
        # runs the backend concurrently. The TUM chain sets "thread_local" in
        # its YAML; without this the two sequences would be capturing under
        # different rules and only one of them would be the tested one.
        if "ITERGRAPH_ERR" in os.environ:
            _ig_cfg["capture_error_mode"] = os.environ["ITERGRAPH_ERR"]
        if "ITERGRAPH_WARMUP" in os.environ:
            _ig_cfg["warmup_iters"] = int(os.environ["ITERGRAPH_WARMUP"])
        self.iter_graph = TrackingIterationGraph(_ig_cfg)
        # PER-ITERATION TRACE (ITER_TRACE= or Training.iter_trace), for
        # profiling/plot_iter_trace.py. The hooks are host Python inside the
        # iteration closure, so a captured replay would record nothing -
        # refused rather than traced wrong.
        self.iter_trace = IterTrace(_train.get("iter_trace", {}))
        if self.iter_trace.enabled and self.iter_graph.enabled:
            raise ValueError(
                "ITER_TRACE cannot run with the iteration graph: its hooks "
                "are host Python inside the captured iteration. Set ITERGRAPH=0.")

        # FULL-MATRIX POSE PRECONDITIONER, ported from SplaTAM.
        #
        # MonoGS IS THE EASIER PORT and this is why: cam_rot_delta and
        # cam_trans_delta ARE the SE(3) tangent, so their .grad is already the
        # 6-vector the method wants. SplaTAM needed quat_trans_grad_to_tangent
        # to map a 7-parameter (q, t) gradient across; here there is nothing to
        # map. tau order is [rot(3), trans(3)] - the REVERSE of SplaTAM's
        # [rho, theta] - which does not matter for M (it is estimated from
        # whatever order it is fed, consistently) but WOULD matter for
        # bfgs_b0's diagonal seed and for se3_adjoint transport. Both are off.
        #
        # THE TUM RESULT NEEDS ALL FOUR PIECES, not just the preconditioner:
        # on fr1_desk the preconditioner with plateau stopping alone was 1/3,
        # and 6/6 once the handoff was added. See results/preconditioner_ladder.
        _ps_enabled_hint = bool(
            _train.get("pixel_sample", {}).get("enabled", False))
        _pre = dict(_train.get("preconditioner", {}))
        # ENV OVERRIDES, same names SplaTAM uses, so one habit drives both.
        # Sweeps are driven from the shell there and the YAML would otherwise
        # have to be edited per arm - which is exactly how arms end up
        # silently sharing a config.
        _envmap = {
            "PRECOND": ("enabled", lambda v: v not in ("0", "", "false", "False")),
            "AUTO_LR": ("auto_lr", lambda v: v not in ("0", "", "false", "False")),
            "PRE_LR": ("lr", float),
            "PRE_TAU": ("tau", float),
            # Second-moment horizon, ~1/(1-beta2) samples. 0.95 gives 20,
            # which covers two thirds of SplaTAM's ~30-iteration frame but
            # only a fifth of MonoGS's 100. Past that horizon M has forgotten
            # the frame's large early gradients and tracks only the current
            # small ones, so numerator and denominator shrink together and the
            # step stops decaying. SplaTAM exposes PRE_BETA1/PRE_BETA2; MonoGS
            # did not, so it could never be swept here.
            "PRE_BETA1": ("beta1", float),
            "PRE_BETA2": ("beta2", float),
            "PRE_SHRINK": ("shrink", float),
            "CARRY": ("carry", lambda v: v not in ("0", "", "false", "False")),
            # Ordinary frames still carry M; only a new frame flagged by the
            # existing anomaly guard discards the carried metric shape.
            "PRE_ANOM_RESET_M": (
                "reset_m_on_anomaly",
                lambda v: v not in ("0", "", "false", "False"),
            ),
            "STOP_REL": ("stop_rel_grad", float),
            "STOP_MODE": ("stop_mode", str),
            "STOP_PATIENCE": ("stop_patience", int),
            "STOP_IMPROVE": ("stop_improve", float),
            "STOP_MIN": ("stop_min_iters", int),
            "STOP_EVERY": ("stop_check_every", int),
            # Unlike STOP_MIN, which belongs only to the preconditioner's
            # plateau rule, HARD_MIN gates every exit from the tracking loop.
            # This isolates whether native MonoGS convergence is cutting the
            # diagonal refinement tail short.
            "HARD_MIN": ("hard_min_iters", int),
            # Halving the eigh rate is the main lever on the preconditioner's
            # fixed per-iteration cost, which matters far more here than on
            # SplaTAM: MonoGS iterations are ~3x cheaper (4.03 ms vs ~12.5),
            # so the same fixed overhead is 24% rather than 6%.
            "PRE_REFACTOR": ("refactor_every", int),
            "PRE_HANDOFF": ("handoff", int),
            # THE HANDOFF-FREE ARM'S TWO INGREDIENTS, ported from SplaTAM
            # where they took the no-handoff configuration from 2/4 to 14/16
            # over sixteen full-length runs.
            #
            # PRE_MAX_STEP=1.0 is NOT a swept value: |M^{-1/2} m|^2 = m^T M^-1 m
            # <= n, so ref = lr*sqrt(n) is the bound the estimator satisfies by
            # construction - the same inequality that bounds Adam's step at lr
            # per coordinate. The shipped default sits 10x above it. Measured on
            # SplaTAM it costs no iterations, because the mean step is ~0.22x
            # ref and only ~1% of steps are anywhere near the cap.
            #
            # PRE_RESTART=N zeroes the first moment mid-frame, which is half of
            # what the handoff does at its switch (splatam.py asserts Adam's
            # state is empty there). It carries no per-scene units but N IS an
            # iteration count, so it does not port across budgets unchanged -
            # MonoGS's tracking budget is not SplaTAM's, and 20 was chosen to
            # match a handoff at 20.
            "PRE_MAX_STEP": ("max_step_mult", float),
            "PRE_RESTART": ("restart_at", int),
            "PRE_RESTART_M": ("restart_m", str),
            # Full-matrix acquisition followed by diagonal refinement from
            # the SAME accumulated second moment.  This is the handoff-free
            # replacement validated on SplaTAM: unlike PRE_HANDOFF it never
            # gives the pose block back to Adam, and unlike the normalised-
            # gradient ramp it retains learned per-coordinate scaling.
            "PRE_DIAG_AFTER": ("diag_after", int),
            # No fixed iteration: validate each full proposal with the next
            # render, then latch the diagonal tail on the first non-decrease.
            "PRE_DIAG_ADAPTIVE": (
                "adaptive_diag",
                lambda v: v not in ("0", "", "false", "False"),
            ),
            "PRE_DIAG_PATIENCE": ("adaptive_diag_patience", int),
            "PRE_DIAG_CALIB": ("adaptive_diag_calibration_frames", int),
            # Floor on the CALIBRATED transition. TUM fr1_desk learned D=6 from
            # p10/med/p90=3/6/8, and both the MonoGS fixed-D arm (diag20, ATE
            # 1.489 at full budget) and Gaussian-SLAM's diag20->diag40 result
            # (aligned 3.51 -> 2.59) point the other way: a longer coupled
            # acquisition phase, not a shorter one. This makes that testable
            # without abandoning the calibration.
            "PRE_DIAG_MIN": ("adaptive_diag_min_iter", int),
            # Instrumentation. PRE_PROFILE_EVERY windows the pre-cap step
            # profile; DRIFT_MULT is the step-based guard. Both default off.
            "PRE_PROFILE_EVERY": ("profile_every", int),
            # Raw per-frame IT/FRAME/REL-GRAD/K series in summary(). Off by
            # default - lr_ladder.sh sets it for profiling/lr_proxy_replay.py.
            "LR_SERIES": ("log_series",
                          lambda v: v not in ("0", "", "false", "False")),
            "DRIFT_MULT": ("drift_mult", float),
            "DRIFT_WARMUP": ("drift_warmup", int),
            # PRE_DEAD_FREEZE=0 restores the pre-f29bbb7 behaviour, where an
            # empty render still decayed M. On by default and bit-exact on any
            # run without empty renders.
            "PRE_DEAD_FREEZE": ("dead_freeze",
                                lambda v: v not in ("0", "", "false", "False")),
            "FINAL_DENSE": ("final_dense",
                            lambda v: v not in ("0", "", "false", "False")),
        }
        for _k, (_name, _cast) in _envmap.items():
            if _k in os.environ:
                _pre[_name] = _cast(os.environ[_k])
        _pre_stop = {k: _pre.pop(k) for k in
                     ("stop_rel_grad", "stop_min_iters", "stop_check_every",
                      "stop_mode", "stop_patience", "hard_min_iters",
                      "handoff", "final_dense")
                     if k in _pre}
        self.pre_stop_rel = float(_pre_stop.get("stop_rel_grad", 0.0))
        self.pre_stop_min = int(_pre_stop.get("stop_min_iters", 10))
        self.pre_stop_every = max(1, int(_pre_stop.get("stop_check_every", 5)))
        self.pre_stop_mode = str(_pre_stop.get("stop_mode", "rel"))
        self.pre_stop_patience = int(_pre_stop.get("stop_patience", 10))
        self.pre_hard_min = int(_pre_stop.get("hard_min_iters", 0))
        self.pre_handoff = int(_pre_stop.get("handoff", 0))
        self.pre_final_dense = bool(_pre_stop.get("final_dense", False))
        if not 0 <= self.pre_hard_min <= self.tracking_itr_num:
            raise ValueError(
                "preconditioner hard_min_iters must be between 0 and "
                f"tracking_itr_num ({self.tracking_itr_num}), got "
                f"{self.pre_hard_min}"
            )
        # A floor at or above the budget would install a transition the frame
        # can never reach, silently turning the arm into a full-matrix control
        # that still reports itself as a diagonal-tail run.
        _diag_min = int(_pre.get("adaptive_diag_min_iter", 0))
        if not 0 <= _diag_min < self.tracking_itr_num:
            raise ValueError(
                "preconditioner adaptive_diag_min_iter must be between 0 and "
                f"tracking_itr_num-1 ({self.tracking_itr_num - 1}), got "
                f"{_diag_min}"
            )
        self.pose_pre = None
        if _pre.pop("enabled", False):
            _pre.pop("dim", None)
            # lr from the preconditioner block, NOT Training.lr - the scales
            # are not interchangeable. AUTO_LR derives it from MonoGS's own
            # tuned deltas instead, which is what makes it portable across
            # datasets; SplaTAM's ladder records PRE_LR=0.004 carried blind to
            # a new scene giving 17.49 cm against Adam's 0.24.
            _auto = _pre.pop("auto_lr", False)
            _tgt = float(_pre.pop("target_step_norm", 0.0))
            _lr_r = self._cam_lr("cam_rot_delta")
            _lr_t = self._cam_lr("cam_trans_delta")
            if _auto and _tgt <= 0.0:
                _tgt = (3.0 * _lr_t ** 2 + 3.0 * _lr_r ** 2) ** 0.5
                Log(f"[Preconditioner] auto step norm {_tgt:.3e} from "
                    f"cam_trans_delta={_lr_t:g}, cam_rot_delta={_lr_r:g}")
            # AUTO_LR_SCALE - THE KNOB THE HANDOFF SAYS TO ADD FIRST, and the
            # reason is that the formula above IS the rule the record already
            # killed on the other two models. sqrt(3 lr_t^2 + 3 lr_r^2) fitted
            # k = 0.57 twice on Gaussian-SLAM and 0.82 (TUM) against 0.28
            # (Replica) on SplaTAM - every fitted k BELOW one - while MonoGS
            # inherits it at k = 1 with no way to test that.
            #
            # This is the same inheritance that turned out to be wrong on
            # Gaussian-SLAM/Replica, where halving cam_trans_lr took room0 from
            # 0.69 to 0.17 cm AND ran faster.
            #
            # READ IT AGAINST it/frame, NOT ALONE. MonoGS's native convergence
            # is `tau.norm() < 1e-4` - a FIXED ABSOLUTE threshold - while the
            # preconditioner PINS the acquisition step to this norm. Scaling
            # the norm down scales the whole step profile down, so the fixed
            # threshold is crossed sooner and it/frame falls MECHANICALLY,
            # whether or not the pose is better converged. A shorter frame is
            # therefore not evidence on its own; the pair (ATE, it/frame) is.
            _lr_scale = float(os.environ.get("AUTO_LR_SCALE", "1") or 1.0)
            if _lr_scale != 1.0:
                if _lr_scale <= 0.0:
                    raise ValueError(
                        f"AUTO_LR_SCALE must be > 0 (got {_lr_scale})")
                _tgt *= _lr_scale
                Log(f"[Preconditioner] AUTO_LR_SCALE={_lr_scale:g} -> step "
                    f"norm {_tgt:.3e}. Native convergence is a FIXED "
                    f"tau.norm() < 1e-4, so a smaller step shortens frames by "
                    f"construction - read ATE against it/frame, not it/frame "
                    f"alone.")
            if (_auto and (int(_pre.get("diag_after", 0)) > 0
                           or bool(_pre.get("adaptive_diag", False)))
                    and "diag_lr" not in _pre):
                # Keep the established fixed-norm full-matrix acquisition,
                # then recover MonoGS Adam's actual coordinate-wise scale in
                # the diagonal tail. Without this, target_step_norm pins every
                # post-switch proposal to the same 0.005477 norm and the tail
                # cannot refine; the 100-frame smoke run showed 1.00x in every
                # iteration band and stopped at 30.1 iters/frame.
                _pre["diag_lr"] = [_lr_r] * 3 + [_lr_t] * 3
                Log(f"[Preconditioner] diagonal-tail rates "
                    f"rot={_lr_r:g}, trans={_lr_t:g}")
            # ONLINE_LR_TUNE - the within-run alternative to AUTO_LR_SCALE.
            # AUTO_LR_SCALE fixes k for the whole run from an offline probe
            # (profiling/lr_probe.py) run beforehand; this discovers k from
            # this run's own it/frame instead, by alternating k and k/2 on
            # adjacent frames (12-24 frames per rung) and then holding fixed
            # for the rest of the scene. See utils/online_lr_tuner.py. The two
            # are mutually exclusive - ONLINE_LR_TUNE starts its own search
            # from k=1.0 regardless of any AUTO_LR_SCALE already applied to
            # _tgt above, since it needs a known reference to halve from.
            _online_tune = bool(int(os.environ.get("ONLINE_LR_TUNE", "0")))
            if _online_tune:
                Log("[Preconditioner] ONLINE_LR_TUNE=1 - target_step_norm "
                    "will be discovered from this run's own it/frame over "
                    "its first frames, then held fixed.")
            self.pose_pre = PosePreconditioner(
                dim=6, lr=float(_pre.pop("lr", _lr_t)),
                target_step_norm=_tgt, device="cuda",
                online_lr_tune=_online_tune,
                online_lr_block_frames=int(
                    os.environ.get("ONLINE_LR_BLOCK", "12")),
                online_lr_max_halvings=int(
                    os.environ.get("ONLINE_LR_MAX_HALVINGS", "4")),
                online_lr_budget=self.tracking_itr_num,
                online_lr_log_fn=Log,
                **_pre)
            if self.pose_pre.adaptive_diag and self.pre_handoff > 0:
                raise ValueError(
                    "adaptive_diag already owns the phase transition and "
                    "cannot be combined with a handoff")
            if self.iter_graph.enabled:
                # GRAPH + PRECONDITIONER IS SUPPORTED, and the three things
                # that genuinely break it are refused INDIVIDUALLY below. The
                # blanket refusal that used to stand here also barred the
                # configuration the suite actually runs - handoff 0, final_dense
                # off, fixed diagonal - which is the same shape Gaussian-SLAM
                # captures at 2.20x. What made it safe there makes it safe here:
                # pose_pre.step() is branch-free, keeps its state in place, and
                # carries a device-side bias counter, while refactor() (the
                # eigh) runs OUTSIDE the captured region. The fixed diagonal
                # transition is NOT a second kind of iteration - it is a step
                # BLEND weighted by _diag_w, a device scalar that refactor()
                # fills in place, so the replayed kernels read the current
                # weight without a recapture.
                #
                # THE HANDOFF really is two kinds of iteration - preconditioned
                # then Adam - and Gaussian-SLAM pays for it with phase-keyed
                # capture. MonoGS's call site passes no phase, so rather than
                # replay the wrong kernels for the tail of every frame, refuse.
                if self.pre_handoff > 0:
                    raise ValueError(
                        "preconditioner handoff cannot run with "
                        "iteration_graph: iterations 0..handoff-1 take the "
                        "preconditioned update and the rest take Adam's, and "
                        "this call site captures one phase per frame. Run with "
                        "PRE_HANDOFF=0 or ITERGRAPH=0.")
                # final_dense switches the RENDER WORKLOAD mid-frame (one
                # full-resolution gradient iteration at the stopping point),
                # which changes tensor shapes a capture has already recorded.
                if self.pre_final_dense:
                    raise ValueError(
                        "final_dense cannot run with iteration_graph: it "
                        "swaps in a full-resolution iteration at the stopping "
                        "point and a capture freezes the shapes it recorded. "
                        "Run with FINAL_DENSE=0 or ITERGRAPH=0.")
                # The adaptive transition reads the frame's own loss to the
                # host BEFORE applying each acquisition step - a sync, illegal
                # inside a capture, and it would in any case read a replayed
                # buffer. Same refusal Gaussian-SLAM's tracker carries.
                if self.pose_pre.adaptive_diag:
                    raise ValueError(
                        "PRE_ADAPTIVE_DIAG needs a host read of the loss every "
                        "acquisition iteration, which cannot happen inside a "
                        "CUDA graph capture. Run with DIAG=<n> (fixed) or "
                        "ITERGRAPH=0. The graph changes execution and not "
                        "arithmetic, so an ATE measured without it is still "
                        "comparable.")
            Log(f"[Preconditioner] {self.pose_pre.summary()}")
            if self.pre_hard_min > 0:
                Log(f"[Preconditioner] hard tracking floor: "
                    f"{self.pre_hard_min} completed iterations; gates native "
                    "convergence, loss stopping, and preconditioner plateau")
            # SAY WHICH STOPPING RULE IS ACTUALLY LIVE. Three can be on at
            # once here - MonoGS's native `converged`, the loss-based
            # EarlyStop, and this - and a silent one is how an arm ends up
            # measuring something other than what it claims.
            if self.pre_stop_rel > 0.0 and self.pre_stop_mode == "best":
                Log(f"[Preconditioner] PLATEAU stopping: no new best |g| for "
                    f"{self.pre_stop_patience} iters (improve margin "
                    f"{self.pose_pre.stop_improve:g}), floor "
                    f"{self.pre_stop_min}, checked every "
                    f"{self.pre_stop_every}")
            elif self.pre_stop_rel > 0.0:
                Log(f"[Preconditioner] gradient-norm stopping: |g|/|g0| < "
                    f"{self.pre_stop_rel} after >= {self.pre_stop_min} iters")
            else:
                Log("[Preconditioner] no preconditioner stopping rule "
                    "(stop_rel_grad = 0)")
            if self.pre_final_dense and not _ps_enabled_hint:
                Log("[Preconditioner] final_dense is ON but pixel_sample is "
                    "OFF - it can only pay when tile masking is on, so it is "
                    "inert here")
            if self.pre_handoff > 0:
                Log(f"[Handoff] preconditioner for iters 0-{self.pre_handoff - 1}, "
                    f"then Adam")
        # MonoGS optimises a 6-vector (cam_rot_delta(3) + cam_trans_delta(3)),
        # not SplaTAM's quaternion+xyz, and needs one extra column to carry
        # update_pose's `converged` flag out of the captured region.
        _es_cfg = dict(_train.get("es_signals", {}))
        _wconv_requested = os.environ.get("WCONV", "0") not in (
            "0", "", "false", "False"
        )
        if _wconv_requested:
            _es_cfg["enabled"] = True
            _es_cfg["batch"] = int(os.environ.get("WCONV_BATCH", "8"))
        self.es_signals = ESSignalBuffer(_es_cfg, pose_dim=6, extra_cols=1)
        # GRADIENT REUSE. Same flags as SplaTAM and Gaussian-SLAM -
        # GRAD_REUSE, GRAD_REUSE_WARMUP, GRAD_REUSE_NOACC - so one sweep
        # drives all three and an arm cannot be on for one model and off
        # for another without the run name saying so.
        #
        # WARMUP IS A BUDGET FRACTION, NOT A COUNT, AND 6 IS A SplaTAM
        # NUMBER. It protects ~23% of a 26-iteration SplaTAM frame and
        # only ~9% of MonoGS's ~69, so carrying the integer across carries
        # a different setting. Expect it to need scaling here, and use
        # GRADSTALE rather than ATE to find the right value.
        _gr_cfg = grad_reuse_env(_train.get("grad_reuse", {}))
        _gr_cal_frames = int(os.environ.get(
            "GRAD_REUSE_CALIBRATE_FRAMES", "0") or 0
        )
        if _gr_cal_frames:
            # The full-matrix acquisition needs fresh gradients. Reusing
            # inside it (especially with FREEZE_M) changes the matrix being
            # learned, which is exactly what the office0 D20 comparison
            # exposed. In calibrated mode this relationship is a rule, not a
            # second knob: warmup always equals the fixed DIAG transition.
            _diag_after = int(getattr(self.pose_pre, "diag_after", 0) or 0)
            if self.pose_pre is None or _diag_after <= 0:
                raise ValueError(
                    "GRAD_REUSE_CALIBRATE_FRAMES requires a fixed positive "
                    "preconditioner diag_after (set DIAG/PRE_DIAG_AFTER)"
                )
            if not bool(_gr_cfg.get("enabled", False)):
                raise ValueError(
                    "GRAD_REUSE_CALIBRATE_FRAMES requires GRAD_REUSE>=2"
                )
            if ("GRAD_REUSE_WARMUP" in os.environ
                    and int(os.environ["GRAD_REUSE_WARMUP"]) != _diag_after):
                Log("[GradReuse] calibrated mode overrides "
                    f"GRAD_REUSE_WARMUP={os.environ['GRAD_REUSE_WARMUP']} "
                    f"with DIAG={_diag_after}")
            if "GRAD_REUSE_COOLDOWN" in os.environ:
                Log("[GradReuse] calibrated mode ignores the fixed "
                    f"GRAD_REUSE_COOLDOWN={os.environ['GRAD_REUSE_COOLDOWN']}; "
                    "the first-frame stopping distribution selects it")
            _gr_cfg["warmup"] = _diag_after
            _gr_cfg["cooldown"] = 0
            _gr_cfg["calibration_frames"] = _gr_cal_frames
            _gr_cfg["calibration_margin"] = int(os.environ.get(
                "GRAD_REUSE_CALIBRATE_MARGIN", "15") or 15
            )
            _gr_cfg["calibration_round"] = int(os.environ.get(
                "GRAD_REUSE_CALIBRATE_ROUND", "10") or 10
            )
            Log("[GradReuse] automatic window calibration: reuse OFF for "
                f"the first {_gr_cal_frames} tracked frames; warmup="
                f"DIAG={_diag_after}; cooldown=floor((median stop - "
                f"{_gr_cfg['calibration_margin']}) / "
                f"{_gr_cfg['calibration_round']}) * "
                f"{_gr_cfg['calibration_round']}")
        self.grad_reuse = GradReuse(_gr_cfg)
        if (self.grad_reuse.enabled and self.iter_graph.enabled
                and self.iter_graph.warmup_iters > self.grad_reuse.warmup):
            # The two compose via iter_graph.run(bypass=...) - reuse
            # iterations run eager, rendering ones replay - but the
            # CAPTURE must still land on a rendering iteration. Bypass
            # returns before _warmup_done is touched, so the graph only
            # ever warms up on rendering iterations; it just needs enough
            # of them before the reuse band opens.
            raise ValueError(
                "iteration_graph.warmup_iters (%d) must be <= "
                "grad_reuse.warmup (%d), or the graph would still be "
                "warming up when reuse starts"
                % (self.iter_graph.warmup_iters, self.grad_reuse.warmup))
        # GRAD_REUSE_STOP_CLOCK=render: the WINDOWED stopper reads only rendered
        # iterations, numbered by render count (see utils.grad_reuse.RenderClock).
        # Inert without reuse or without WCONV. Native tau is unaffected.
        self._render_clock = (
            self.grad_reuse.enabled
            and os.environ.get("GRAD_REUSE_STOP_CLOCK", "") == "render")
        if self._render_clock:
            Log("[Tracking] windowed stopper runs on the RENDER clock "
                "(reused iterations are skipped and their steps banked)")
        _wc = dict(_train.get("windowed_convergence", {}))
        _wc_env = {
            "WCONV": ("enabled", lambda v: v not in ("0", "", "false", "False")),
            "WCONV_SHADOW": ("shadow", lambda v: v not in ("0", "", "false", "False")),
            "WCONV_WINDOW": ("window", int),
            "WCONV_EVERY": ("check_every", int),
            "WCONV_PATIENCE": ("patience", int),
            "WCONV_KIND": ("kind", str),
            "WCONV_POSE_CHANGE": ("pose_change", float),
            "WCONV_LOSS_CHANGE": ("loss_change", float),
            "WCONV_PROGRESS": ("progress_min", float),
            "WCONV_STALL": ("stall_patience", int),
            "WCONV_AFTER_PHASE": ("proposal_after_phase", int),
            "WCONV_PLATEAU_FALLBACK": ("plateau_fallback", lambda v: v not in ("0", "", "false", "False")),
            "WCONV_Z": ("z_threshold", float),
            "WCONV_DECAY": ("decay_ratio", float),
            "WCONV_ENERGY_WINDOW": ("energy_window", int),
            "WCONV_ENERGY_PATIENCE": ("energy_patience", int),
            "WCONV_AUTO": ("auto_enabled", lambda v: v not in ("0", "", "false", "False")),
            "WCONV_AUTO_KIND": ("auto_kind", str),
            "WCONV_AUTO_CALIB": ("auto_calibration_frames", int),
            "WCONV_AUTO_AUDIT": ("auto_audit_every", int),
            "WCONV_AUTO_AUDIT_PATIENCE": ("auto_audit_patience", int),
            "WCONV_AUTO_MIN_PROPOSALS": ("auto_min_proposals", int),
            "WCONV_AUTO_LOSS_P90": ("auto_loss_p90", float),
            "WCONV_AUTO_LOSS_MAX": ("auto_loss_max", float),
            "WCONV_AUTO_MOTION_P90": ("auto_motion_p90", float),
            "WCONV_AUTO_MOTION_MAX": ("auto_motion_max", float),
            "WCONV_AUTO_MIN_SAVED": ("auto_min_saved_fraction", float),
            "WCONV_AUTO_AFTER_PHASE": ("auto_proposal_after_phase", int),
            "WCONV_AUTO_SPEC": ("auto_spec", str),
            "WCONV_AUTO_REPORT": ("auto_report_candidates", lambda v: v not in ("0", "", "false", "False")),
        }
        for _key, (_name, _cast) in _wc_env.items():
            if _key in os.environ:
                _wc[_name] = _cast(os.environ[_key])
        _lr_r = self._cam_lr("cam_rot_delta")
        _lr_t = self._cam_lr("cam_trans_delta")
        _wc["budget"] = int(self.tracking_itr_num)
        # ESSignalBuffer receives [rotation | translation] deltas here, while
        # the shared SE(3) convention is [translation | rotation].  Energy-only
        # rules are order-agnostic; incumbent pose composition is not.
        _wc["auto_step_order"] = "rotation_translation"
        _wc["step_order"] = "rotation_translation"
        self.windowed_stop = make_windowed_convergence(
            _wc, scales=[_lr_r] * 3 + [_lr_t] * 3
        )
        self.windowed_or_native = os.environ.get(
            "WCONV_OR_NATIVE", "0") not in ("0", "", "false", "False")
        self.windowed_plateau_fallback = bool(
            _wc.get("plateau_fallback", False)
        )
        if self.windowed_plateau_fallback and not (
                self.windowed_stop.active
                and isinstance(self.windowed_stop, IncumbentConvergence)):
            raise ValueError(
                "WCONV_PLATEAU_FALLBACK requires active direct incumbent "
                "convergence"
            )
        self.windowed_sweep = WindowedConvergenceSweep(
            os.environ.get("WCONV_SWEEP", ""),
            _wc,
            scales=[_lr_r] * 3 + [_lr_t] * 3,
        )
        if self.windowed_stop.enabled and not self.es_signals.enabled:
            raise ValueError(
                "windowed_convergence requires es_signals.enabled: its loss "
                "and full tangent update must come from the same iteration"
            )
        if (self.grad_reuse.calibration_frames
                and not self.windowed_stop.active):
            raise ValueError(
                "GRAD_REUSE_CALIBRATE_FRAMES requires an active WCONV "
                "stopping criterion during its run-in frames"
            )
        if (self.windowed_stop.active
                and self.windowed_stop.check_every % self.es_signals.batch != 0):
            raise ValueError(
                "active windowed_convergence check_every must be a multiple "
                "of es_signals.batch so its decision lands on a drain boundary"
            )
        Log(f"[Tracking] {self.windowed_stop.summary()}")
        Log(f"[Tracking] {self.grad_reuse.summary()}")
        for _line in self.windowed_sweep.summary_lines():
            Log(f"[Tracking] {_line}")
        # CAPTURE_SAFE_POSE=1 turns the third arm on without editing the YAML,
        # and it exists because the graph FORCES it: _preflight_iteration_graph
        # calls set_capture_safe_pose(True) unconditionally. The TUM chain sets
        # it in fr1_desk_posefix_async.yaml, so a TUM graph A/B is matched by
        # accident. THE REPLICA CHAIN SETS IT NOWHERE, so a Replica graph arm
        # would carry both the capture AND three deleted per-render
        # torch.linalg.inv calls against a control carrying neither - and the
        # inv calls are in MAPPING's renders too, which no tracking-side change
        # can reach. That is precisely how the graph took credit for an
        # unrelated fix once already; the control needs this flag to be a
        # control at all.
        _csp = bool(_train.get("capture_safe_pose", False))
        if "CAPTURE_SAFE_POSE" in os.environ:
            _csp = os.environ["CAPTURE_SAFE_POSE"] not in (
                "0", "", "false", "False")
        if self.iter_graph.enabled:
            self._preflight_iteration_graph(_train)
        elif _csp:
            # THE THIRD ARM. Measured on the 100-frame matched-work A/B:
            #
            #   tracking   37.3s vs 51.8s   (4.75 vs 6.59 ms/iter)   1.39x
            #   kf_logic  108.2s vs 144.5s                           1.34x
            #   total     164.2s vs 207.1s                           1.26x
            #
            # kf_logic is MAPPING, and a tracking-side change cannot touch
            # mapping - it is the control variable. It moved by 36.3s, MORE
            # than the 14.5s tracking saved, because set_capture_safe_pose was
            # called only on the graph arm and it deletes three
            # torch.linalg.inv calls from every render, mapping's included.
            #
            # So the wall-time win was mostly NOT the CUDA graph. This flag
            # exists to attribute it: capture-safe pose ON, graph OFF. Without
            # it the two effects are permanently welded together and the graph
            # gets credit for an unrelated fix.
            set_capture_safe_pose(True)
            Log("[Tracking] capture-safe pose ON, iteration graph OFF - the "
                "third arm. Three per-render torch.linalg.inv calls are gone "
                "from tracking AND mapping.")

    def _preflight_iteration_graph(self, train_cfg):
        """Check the graph's prerequisites at startup, not at frame 300.

        Every item here is something that produces either a hard capture
        failure or - worse - a run that captures 100% of frames and silently
        computes the wrong thing. The failure modes are called out individually
        because the whole cost of the SplaTAM port was spent distinguishing
        them from each other.
        """
        # The pose path. Without this, update_RT rebinds camera.R/.T to new
        # tensors and the captured render keeps reading the pre-capture pose:
        # the frame would capture cleanly, run its full iteration count, and
        # optimise nothing. See camera_utils.set_capture_safe_pose.
        set_capture_safe_pose(True)

        if not self.binning_capacity.enabled:
            raise ValueError(
                "Training.iteration_graph.enabled requires "
                "Training.binning_capacity.enabled. Without a fixed capacity "
                "the rasterizer reads num_rendered back to the host to size "
                "its binning buffer, which is illegal during capture - the "
                "first capture attempt will fail."
            )

        # The capacity must be frozen BEFORE the capture iteration runs.
        # BinningCapacity returns {"binning_count_out": ...} - the ORIGINAL
        # host-readback path - until it has seen warmup_iters iterations, so if
        # its warmup outlasts the graph's, the capture records the very
        # readback the fixed capacity exists to remove and fails.
        #
        # Gaussian-SLAM's preflight had this check from the start; MonoGS's did
        # not, and the two ports are meant to enforce the same prerequisites.
        # It never fired in practice only because every config here happens to
        # set 3 against 5.
        if self.binning_capacity.warmup_iters > self.iter_graph.warmup_iters:
            raise ValueError(
                f"Training.binning_capacity.warmup_iters "
                f"({self.binning_capacity.warmup_iters}) must be <= "
                f"Training.iteration_graph.warmup_iters "
                f"({self.iter_graph.warmup_iters}). The capacity has to be "
                f"frozen before the capture iteration runs, or the capture "
                f"records the host readback path it is supposed to be "
                f"replacing."
            )

        # The tile mask must not change WITHIN a frame.
        #
        # is_sparse_phase() returns True only for iterations in
        # [full_start_ratio, full_end_ratio), so a window like the default
        # [0.2, 0.8) hands the render a mask for the middle of the frame and
        # None at both ends. The graph captures ONE iteration and replays it, so
        # whichever of those two the capture iteration happened to be is frozen
        # for the rest of the frame - and since capture happens at
        # warmup_iters (early), that is the unmasked variant, silently making
        # tile masking inert while it still logs as engaged.
        #
        # SplaTAM never hit this only because its tuned config uses an
        # always-sparse [0.0, 1.0] window, where the mask is constant by
        # construction. That is an accident of tuning, not a guarantee, so it is
        # checked here rather than assumed.
        _ps = train_cfg.get("pixel_sample", {})
        if _ps.get("enabled", False):
            _start = _ps.get("full_start_ratio", 0.2)
            _end = _ps.get("full_end_ratio", 0.8)
            if _start != 0.0 or _end < 1.0:
                raise ValueError(
                    f"Training.iteration_graph.enabled requires a constant tile "
                    f"mask across each frame, but pixel_sample's window is "
                    f"[{_start}, {_end}) - the mask changes mid-frame and a "
                    f"captured iteration would freeze one phase of it. Use "
                    f"full_start_ratio=0.0, full_end_ratio=1.0 (always sparse), "
                    f"or disable pixel_sample."
                )

        if train_cfg.get("coarse_to_fine", {}).get("enabled", False):
            raise ValueError(
                "Training.iteration_graph.enabled is incompatible with "
                "Training.coarse_to_fine.enabled. Coarse-to-fine swaps "
                "viewpoint.original_image/depth for downsampled arrays "
                "mid-frame, which changes the render's tensor shapes; a graph "
                "records one set of shapes per frame. Turn one of them off, or "
                "recapture per scale level."
            )

        if self.iter_graph.capture_error_mode == "global":
            # Not fatal - it may work - but this is the risk the SplaTAM port
            # never had to face, so it should never be hit by accident.
            print(
                "[FrontEnd] WARNING: iteration_graph.capture_error_mode is "
                "'global' while the backend may be running concurrently in the "
                "same CUDA context. 'global' errors on ANY legacy-default-"
                "stream operation anywhere in the process during capture, "
                "including the backend's. Use 'thread_local' unless you are "
                "deliberately testing this.",
                flush=True,
            )

        if self.stopper.enabled:
            _floor = getattr(self.stopper, "min_iters", None)
            if _floor is not None and _floor <= self.iter_graph.warmup_iters:
                print(
                    f"[FrontEnd] WARNING: early stopping can fire at iteration "
                    f"{_floor}, at or before the graph's warmup of "
                    f"{self.iter_graph.warmup_iters}. Frames that stop that "
                    f"early never reach a capture, so the graph does nothing "
                    f"for them and frames_captured will under-report.",
                    flush=True,
                )
            # min_iters is 70 against SplaTAM/TUM's 200-iteration budget - 35%.
            # MonoGS's budget is tracking_itr_num, typically 100, so the same
            # fraction is ~35, not 70. Set too high and early stopping silently
            # never fires while still logging enabled=True.
            if _floor is not None and _floor >= self.tracking_itr_num:
                raise ValueError(
                    f"Training.early_stop.min_iters={_floor} is >= "
                    f"tracking_itr_num={self.tracking_itr_num}, so early "
                    f"stopping can never fire. Rescale min_iters for this "
                    f"model's iteration budget (and retune_min_iters_floor "
                    f"with it)."
                )

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def _flush_signal_trace(self, rows, reason):
        path = self._signal_trace_path
        new = os.path.getsize(path) == 0
        with open(path, "a") as f:
            if new:
                f.write("frame,iter,loss,step_norm,reused,conv,energy_ratio,"
                        "proposed,reason\n")
            for r in rows:
                f.write("%d,%d,%.9g,%.9g,%d,%.0f,%.6g,%d,%s\n" % (*r, reason))

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self._cam_lr("cam_rot_delta"),
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self._cam_lr("cam_trans_delta"),
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        # capturable=True is REQUIRED for the iteration graph, and its absence
        # is not a graceful degradation: Adam's step raises "If capturable=False,
        # state_steps should not be CUDA tensors" the moment capture starts,
        # because a capturable Adam keeps `step` on the device instead of the
        # host. Tied to the graph flag so the non-graph arm keeps the original
        # (slightly cheaper) code path and the A/B stays honest.
        pose_optimizer = build_tracking_optimizer(
            opt_params, tracking=True, capturable=self.iter_graph.enabled
        )

        # Tile subsampling: build mask once per frame, reuse across sparse-
        # phase iterations. Full-pixel at the start (pose still coarse) and
        # end (final refinement) of each frame's tracking; sparse only in
        # the middle window (full_start_ratio, full_end_ratio).
        _ps_cfg = self.config.get("Training", {}).get("pixel_sample", {})
        # PS_RATIO / PS_START / PS_END override the YAML, so MonoGS can be run
        # at SplaTAM's tuned shape without a second set of configs.
        #
        # WHY THIS IS WORTH A KNOB. The two models ship DIFFERENT shapes for
        # what is meant to be one shared optimisation: MonoGS keeps 60% of
        # tiles over the middle [0.2, 0.8) of the budget, SplaTAM keeps 75%
        # over the whole frame [0.0, 1.0). They happen to remove almost the
        # same TOTAL tile work - ~26% against ~25% - and SplaTAM gains 10% of
        # tracking ms/iter from it while MonoGS gains 0.00 across five cells.
        # So the difference is not the shape, but "one configuration across
        # three models" is not literally true until they match.
        #
        # [0.0, 1.0) ALSO REMOVES A REAL BEHAVIOURAL DIFFERENCE, not just a
        # cosmetic one. is_sparse_phase divides by tracking_itr_num - the CAP -
        # so with a narrow window and native convergence stopping at ~65 of
        # 100, the frame ends INSIDE the sparse window and the "full at the
        # end" segment never runs. An always-sparse window makes the mask
        # constant per frame and independent of where stopping lands.
        if _ps_cfg:
            _ps_cfg = dict(_ps_cfg)
            # PS_GRADFRAC carried here for parity with the GSLAM tracker -
            # see the note there on why gradient-ranked selection sharpens the
            # loss surface rather than merely shrinking it.
            for _k, _e in (("sample_ratio", "PS_RATIO"),
                           ("full_start_ratio", "PS_START"),
                           ("full_end_ratio", "PS_END"),
                           ("gradient_frac", "PS_GRADFRAC")):
                if _e in os.environ:
                    _ps_cfg[_k] = float(os.environ[_e])
        _tile_mask = None
        if _ps_cfg.get("enabled", False):
            _H = viewpoint.original_image.shape[1]
            _W = viewpoint.original_image.shape[2]
            _tile_mask = build_tile_mask(viewpoint.original_image, _H, _W, _ps_cfg)

        # Coarse-to-fine: pyramid resolution schedule across tracking
        # iterations. Renders at a genuinely reduced resolution for the
        # coarse window, so the CUDA rasterizer launches fewer thread-
        # blocks - unlike pixel_sample's tile masking, which launches the
        # same full grid and skips work inside it (confirmed not to move
        # ms/iter at all). Tuned config (scale=2, coarse_ratio=0.3) puts
        # the coarse window entirely inside iters [0, 30), well before
        # early_stop's min_iters=50 floor - firing mid-coarse shouldn't be
        # reachable at that floor, but the loop still defends against it
        # below in case min_iters is ever lowered again.
        _ctf_cfg = self.config.get("Training", {}).get("coarse_to_fine", {})
        _ctf_enabled = _ctf_cfg.get("enabled", False)
        _ctf_levels_data = {}
        _ctf_current_scale = 1
        if _ctf_enabled:
            _orig_h, _orig_w = viewpoint.image_height, viewpoint.image_width
            _orig_image = viewpoint.original_image
            _orig_depth = viewpoint.depth
            _orig_grad_mask = viewpoint.grad_mask
            for _s, _ in get_levels(_ctf_cfg):
                if _s not in _ctf_levels_data:
                    _h, _w = _orig_h // _s, _orig_w // _s
                    _ctf_levels_data[_s] = {
                        'image': downsample_image(_orig_image, _s),
                        'depth': downsample_depth(
                            torch.from_numpy(_orig_depth).float(), _s
                        ).numpy(),
                        'grad_mask': (
                            F.interpolate(_orig_grad_mask.float().unsqueeze(0),
                                          size=(_h, _w), mode='nearest').squeeze(0)
                            if _orig_grad_mask is not None else None
                        ),
                        'h': _h, 'w': _w,
                    }

        self.stopper.reset()
        if self.pose_pre is not None:
            # CARRIES M, RESETS m and the plateau state. transport=None: MonoGS
            # has no adjoint wired here, and the SplaTAM ladder measured the
            # transport correction as a NULL at n=5 anyway - the carry matters,
            # the correction to it does not.
            # WINDOWED PROFILE, before carry_frame resets the frame state.
            # A cumulative profile averages the frames where the tracker breaks
            # into the hundreds where it does not; the whole point of the
            # window is to see WHERE the distribution changes.
            if self.pose_pre.profile_due():
                Log(f"[Preconditioner] {self.pose_pre.summary()}")
                _sp = self.pose_pre.step_profile()
                if _sp:
                    Log(_sp)
                self.pose_pre.reset_profile()
            self.pose_pre.carry_frame(transport=None)
        _force_dense = False
        self.binning_capacity.reset_for_frame()
        self.iter_graph.reset_for_frame()
        self.es_signals.reset_for_frame()
        _tail_first = _tail_ref_loss = _tail_best = None
        _tail_best_it = _tail_final = None
        self.windowed_stop.reset_frame()
        self.windowed_sweep.reset_frame()
        pose_delta_norm = 1.0
        _pose_delta_ema = None
        _pose_ema_alpha = 0.2  # ~5-iter memory, matches early_stop's patience window

        # Batched-signal state, mirroring SplaTAM's. The EMA is applied on the
        # HOST from the drained rows, exactly as the eager path applies it to
        # the freshly-.item()'d value - only the frequency of the read changes,
        # never the arithmetic.
        _stop_now = False
        _converged_now = False
        _loss_stop_now = False
        _window_stop_now = False
        _pending_stop_reason = None
        _frame_stop_reason = "cap"

        def _consume(rows):
            """Feed drained iterations to the stopper. Returns True to stop.

            Rows after the firing one are dropped: they were executed (the stop
            is detected up to batch-1 iterations late) but the eager path would
            never have run them, so ignoring them keeps the decisions identical
            and makes this a pure performance change.
            """
            nonlocal _pose_delta_ema, _converged_now, _loss_stop_now
            nonlocal _tail_first, _tail_ref_loss, _tail_best
            nonlocal _tail_best_it, _tail_final
            nonlocal _window_stop_now
            for _r in rows:
                if _sig_trace is not None:
                    _tr_it = int(_r["iter"])
                    _sig_trace.append([
                        cur_frame_idx, _tr_it, float(_r["loss"]),
                        float(np.sqrt(sum(float(v) ** 2 for v in
                                          list(_r["rot"]) + list(_r["tran"])))),
                        (1 if _gr_hist[_tr_it] else 0)
                        if _tr_it < len(_gr_hist) else -1,
                        float(_r["extra"][0]), float("nan"), 0])
                if self._tail_diag:
                    # BEFORE any stop check, so a firing row is still counted.
                    _it, _ls = int(_r["iter"]) + 1, float(_r["loss"])
                    if _tail_first is None:
                        _tail_first = _ls
                    if _tail_best is None or _ls < _tail_best:
                        _tail_best, _tail_best_it = _ls, _it
                    if _it <= self._tail_ref:
                        _tail_ref_loss = _ls
                    _tail_final = _ls
                if self.windowed_stop.enabled:
                    _step = np.asarray(
                        list(_r["rot"]) + list(_r["tran"]), dtype=np.float64
                    )
                    _w_it, _w_skip = _r["iter"], False
                    if _clock is not None:
                        _mapped = _clock.map_row(int(_r["iter"]), _step)
                        if _mapped is None:
                            _w_skip = True
                        else:
                            _w_it, _step = _mapped
                    if _w_skip:
                        _primary_proposed = False
                    else:
                        _primary_proposed = self.windowed_stop.observe(
                            _w_it, _r["loss"], _step
                        )
                    if _sig_trace is not None:
                        _sig_trace[-1][6] = float(getattr(
                            self.windowed_stop, "_current_decay_ratio",
                            float("nan")))
                        _sig_trace[-1][7] = int(bool(_primary_proposed))
                    if (_primary_proposed and self.pose_pre is not None
                            and hasattr(self.windowed_stop,
                                        "note_barred_proposal")):
                        _window_bad = (
                            float(self.pose_pre.anomalous()) > 0.5
                            or float(self.pose_pre.drifted()) > 0.5
                        )
                        if _window_bad:
                            self.windowed_stop.note_barred_proposal()
                            _primary_proposed = False
                    if not _w_skip:
                        self.windowed_sweep.observe(
                            _w_it, _r["loss"], _step
                        )
                    if _primary_proposed:
                        _window_stop_now = True
                        if self.windowed_stop.active:
                            return True
                if (self.windowed_stop.active
                        and (not self.windowed_or_native
                             or getattr(
                                 self.windowed_stop,
                                 "collecting_evidence",
                                 False,
                             ))):
                    # Active mode normally OWNS convergence: native tau, loss
                    # early-stop and the gradient plateau are excluded so a
                    # race between them cannot be mistaken for this criterion.
                    # Every recorded measurement used that exclusive mode.
                    #
                    # WCONV_OR_NATIVE=1 makes it an OR after any automatic
                    # controller has completed its full-budget calibration
                    # frame. Evidence frames still exclude native stopping so
                    # every decay candidate sees the complete tail. The OR is
                    # a production choice, not a
                    # measurement one: it can only stop EARLIER than either
                    # alone, and the stop-reason counts stay separable because
                    # a firing row returns immediately, so at most one rule
                    # claims any given row.
                    #
                    # It exists because activating the shared criterion on
                    # MonoGS DISPLACED a better rule: native stopped 559/591
                    # frames at 68.4 iterations/frame, while the incumbent rule
                    # stopped 204 and let 387 reach the cap at 94.6.
                    continue
                # `iter` is zero-based. A floor of 70 permits the first stop
                # only after iteration 69 has completed. Ignore (rather than
                # latch) signals before then, otherwise a batched native
                # convergence signal would fire immediately when the floor
                # opens even if the current updates were no longer converged.
                _floor_open = (
                    self.pose_pre is None
                    or _r["iter"] + 1 >= self.pre_hard_min
                )
                # extra[0] is update_pose's `converged`, cast to float on the
                # device. It gates the loop in the eager path via `if converged`
                # before the early-stop check, so it is checked first here too.
                if _floor_open and _r["extra"][0] > 0.5:
                    _converged_now = True
                    return True
                _raw = _r["pose_delta"]
                _pose_delta_ema = (
                    _raw if _pose_delta_ema is None
                    else _pose_ema_alpha * _raw + (1 - _pose_ema_alpha) * _pose_delta_ema
                )
                if (_floor_open and self.stopper.enabled
                        and self.stopper.check(
                    _r["iter"], _r["loss"], _pose_delta_ema
                )):
                    _loss_stop_now = True
                    return True
            return False

        # NVTX marker so a profiler can isolate TRACKING's rasterizer kernels
        # from MAPPING's. They cannot be told apart by kernel name: MonoGS's
        # render() always passes theta/rho, so compute_pose_grad is true for
        # both and both launch renderCUDABackward<3, true>. Nor by grid size -
        # same resolution. An NVTX range is the only clean separator.
        #
        # NVTX PUSH/POP RANGES ARE THREAD-LOCAL, and autograd runs backward on
        # its own CUDA worker thread. So this range covers the FORWARD kernels
        # but NOT renderCUDABackward - which is the 36.8% kernel we actually
        # care about. Measured: nsys records 29 PushPop instances of this
        # range, and ncu still reports "No kernels were profiled" when
        # filtering renderCUDABackward on it.
        #
        # MONOGS_PROFILE_SYNC_AUTOGRAD=1 makes the autograd engine run backward
        # on the CALLING thread, bringing those launches inside the range. It
        # is profiling-only: it changes how backward is scheduled, so it must
        # never be set for a timing run. Kernel-intrinsic metrics (L2 hit rate,
        # occupancy, registers) are unaffected by which thread launched them.
        if os.environ.get("MONOGS_PROFILE_SYNC_AUTOGRAD") == "1" and hasattr(
            torch.autograd, "set_multithreading_enabled"
        ):
            torch.autograd.set_multithreading_enabled(False)
        torch.cuda.nvtx.range_push("monogs_tracking")
        _window_phase_signature = None
        _adaptive_diag_enabled = (
            self.pose_pre is not None and self.pose_pre.adaptive_diag
        )
        _adaptive_calibration_frame = (
            _adaptive_diag_enabled
            and self.pose_pre.adaptive_diag_calibration_frames > 0
        )
        _adaptive_diag_active = False
        _adaptive_switch_iteration = None
        _adaptive_pending_loss = None
        _adaptive_pending_R = None
        _adaptive_pending_T = None
        _adaptive_pending_objective = None
        _adaptive_failure_count = 0
        if self.grad_reuse.enabled and _adaptive_diag_enabled:
            # THE ADAPTIVE DIAGONAL VALIDATES A PROPOSAL WITH THE NEXT
            # RENDER - that is why it splits render/backward from the
            # update at all. A reuse iteration renders nothing, so every
            # proposal would be scored against a repeat of its own loss,
            # read as a failure, and rolled back on a cadence set by the
            # reuse period rather than by the data.
            #
            # Refused rather than hooked: the two mechanisms want the same
            # iteration for different things.
            raise ValueError(
                "grad_reuse cannot run with adaptive_diag: it validates each "
                "proposal with the next render, and a reuse iteration produces "
                "none")
        # The stash NEVER crosses a frame boundary: between frames the
        # pose jumps to a new initialisation, so a carried gradient would
        # have been evaluated at a pose the optimiser is nowhere near.
        self.grad_reuse.reset_frame()
        _gr_stash = [None]
        _gr_last = [None]
        _gr_reused = [False]
        _gr_hist = []
        _clock = RenderClock() if self._render_clock else None
        _sig_trace = [] if self._signal_trace_path else None
        # Side channel for the adaptive-reuse trust signal: the closure below
        # is replayed from a CUDA graph (no Python runs), so the fresh
        # gradient is handed out by a list write and read after run() returns.
        _gxi_out = [None]
        self.iter_trace.begin_frame(cur_frame_idx, self.tracking_itr_num,
                                    rot_to_quat(viewpoint.R), viewpoint.T)
        if self.iter_trace.active and _adaptive_diag_enabled:
            # The calibrated transition splits render and step into a
            # separate path the hooks do not cover; use a fixed PRE_DIAG_AFTER.
            raise ValueError(
                "ITER_TRACE does not cover the adaptive diagonal path: set "
                "PRE_DIAG_ADAPTIVE=0 with a fixed PRE_DIAG_AFTER.")
        _gr_pose_params = [viewpoint.cam_rot_delta,
                           viewpoint.cam_trans_delta]
        for tracking_itr in range(self.tracking_itr_num):
            _iter_t0 = time.perf_counter()
            # Update viewpoint resolution when the scale level changes
            if _ctf_enabled:
                _new_scale = get_iter_scale(tracking_itr, self.tracking_itr_num, _ctf_cfg)
                if _new_scale != _ctf_current_scale:
                    if _new_scale == 1:
                        viewpoint.original_image = _orig_image
                        viewpoint.depth = _orig_depth
                        viewpoint.grad_mask = _orig_grad_mask
                        viewpoint.image_height = _orig_h
                        viewpoint.image_width = _orig_w
                    else:
                        _lvl = _ctf_levels_data[_new_scale]
                        viewpoint.original_image = _lvl['image']
                        viewpoint.depth = _lvl['depth']
                        viewpoint.grad_mask = _lvl['grad_mask']
                        viewpoint.image_height = _lvl['h']
                        viewpoint.image_width = _lvl['w']
                    _ctf_current_scale = _new_scale
                if _ctf_current_scale != 1:
                    self._ctf_iters_total += 1
            _is_sparse = is_sparse_phase(
                tracking_itr, self.tracking_itr_num, _ps_cfg)
            if _force_dense:
                # THE DENSE PASS AN EARLY STOP WOULD OTHERWISE SKIP. MonoGS
                # runs full resolution at the START and the LAST
                # (1 - full_end_ratio) of each frame - 20% at the shipped
                # 0.2/0.8 - so a frame stopping inside the sparse window skips
                # every one of those iterations, not just one as in SplaTAM.
                _is_sparse = False
            _active_tile_mask = _tile_mask if _is_sparse else None
            if _active_tile_mask is not None:
                self._sparse_iters_total += 1

            # Render, loss, backward, optimizer step and the pose update, as
            # one callable so TrackingIterationGraph can record the whole thing
            # into a CUDA graph. Everything that synchronises or branches on a
            # device value stays outside it, below.
            _bin_kwargs = self.binning_capacity.render_kwargs()
            _objective_signature = (
                _ctf_current_scale,
                _active_tile_mask is not None,
            )
            if (_adaptive_pending_loss is not None
                    and _adaptive_pending_objective != _objective_signature):
                # Losses from different resolutions/masks are not comparable.
                # Accept the prior proposal and establish a fresh baseline in
                # the new objective instead of manufacturing a transition.
                _adaptive_pending_loss = None
                _adaptive_pending_R = None
                _adaptive_pending_T = None
                _adaptive_pending_objective = None
                _adaptive_failure_count = 0
            # Before the handoff the preconditioner drives; after it, Adam.
            _use_pre = (self.pose_pre is not None
                        and (self.pre_handoff <= 0
                             or tracking_itr < self.pre_handoff))
            if self.windowed_stop.enabled:
                if not _use_pre:
                    _update_phase = "adam"
                elif (_adaptive_diag_enabled and _adaptive_diag_active):
                    _update_phase = "pre-diagonal"
                elif (getattr(self.pose_pre, "diag_after", 0) > 0
                      and tracking_itr >= self.pose_pre.diag_after):
                    _update_phase = "pre-diagonal"
                elif (getattr(self.pose_pre, "restart_at", 0) > 0
                      and tracking_itr >= self.pose_pre.restart_at):
                    _update_phase = "pre-restarted"
                else:
                    _update_phase = "pre-full"
                _window_signature = (
                    _ctf_current_scale,
                    _active_tile_mask is not None,
                    _update_phase,
                )
                if _window_phase_signature is None:
                    _window_phase_signature = _window_signature
                elif _window_signature != _window_phase_signature:
                    # Iteration 20 is halfway through batch=8. Consume rows
                    # 16..19 under the OLD phase, then restart the ring at slot
                    # zero before any diagonal row is recorded. Resetting the
                    # statistical window without draining would still mix the
                    # two regimes when rows 16..23 arrived together.
                    _phase_stop = _consume(self.es_signals.drain_phase())
                    _stop_now = _stop_now or _phase_stop
                    _phase_it = (_clock.before(tracking_itr)
                                 if _clock is not None else tracking_itr)
                    self.windowed_stop.start_phase(
                        _phase_it,
                        f"scale={_window_signature[0]},"
                        f"sparse={int(_window_signature[1])},"
                        f"update={_window_signature[2]}",
                    )
                    self.windowed_sweep.start_phase(
                        _phase_it,
                        f"scale={_window_signature[0]},"
                        f"sparse={int(_window_signature[1])},"
                        f"update={_window_signature[2]}",
                    )
                    _window_phase_signature = _window_signature

            # DECIDED HERE, NOT INSIDE THE CLOSURE, because the CALLER
            # needs the answer: a reuse iteration bypasses the CUDA graph,
            # and iter_graph.run() must be told before it would otherwise
            # replay. The closure only READS the flag now.
            #
            # NEVER on the last iteration: this model commits from the
            # render that produced its best loss, and a reuse iteration
            # produces no loss to compare.
            _gr_reused[0] = (
                self.grad_reuse.should_reuse(
                    tracking_itr, tracking_itr >= self.tracking_itr_num - 1)
                and restore_grads(_gr_pose_params, _gr_stash[0]))
            # COUNTED HERE TOO, and this one is a REPORTING bug if left
            # inside: a rendered iteration REPLAYS under the graph, so the
            # Python body never runs and note_rendered() would only count
            # the warmup and the capture.
            # DROP THE PREVIOUS ITERATION BEFORE THE CAPTURE RUNS.
            # _gr_last holds iteration N-1's WHOLE render package, which keeps
            # warmup-era autograd accumulators alive; the capture then reuses
            # nodes stamped with the legacy default stream and backward()
            # fails with "operation would make the legacy stream depend on a
            # capturing blocking stream". Same hazard SplaTAM fixes with
            # `loss = losses = None`, reintroduced here because holding the
            # previous iteration IS gradient reuse.
            #
            # Safe to drop: clearing the stash makes restore_grads() return
            # False, so the capture iteration renders - which is what it has
            # to do anyway.
            if not _gr_reused[0] and self.iter_graph.will_capture():
                _gr_last[0] = None
                _gr_stash[0] = None
            _gr_hist.append(bool(_gr_reused[0]))
            if _clock is not None:
                _clock.note(bool(_gr_reused[0]))
            if _gr_reused[0]:
                self.grad_reuse.note_reused()
            elif self.grad_reuse.enabled:
                # Gated so the counters stay at zero when reuse is off.
                self.grad_reuse.note_rendered()
            def _tracking_iteration(_mask=_active_tile_mask, _bk=_bin_kwargs,
                                    _it=tracking_itr):
                # GRADIENT REUSE. The whole render, loss and backward are
                # skipped - not one stage of them - so preprocess, binning
                # and the sort go with them.
                #
                # THE SKIP MUST COVER zero_grad TOO. It sits between the
                # render and the loss here, so a skip that covered only the
                # render would zero the gradient just restored and the
                # reuse iteration would step on nothing at all - a silent
                # no-op that still costs an iteration.
                #
                # NEVER on the last iteration: this model commits from the
                # render that produced its best loss, and a reuse iteration
                # produces no loss to compare.
                if _gr_reused[0]:
                    # _pkg TOO, not just the three tensors read out of
                    # it. The closure RETURNS _pkg, and the caller reads
                    # depth and opacity from it for get_median_depth and
                    # the keyframe decision - leaving it unbound raised
                    # UnboundLocalError on the first reuse iteration.
                    #
                    # It is one step stale on a reuse iteration, and that
                    # staleness is bounded by exactly the quantity
                    # convergence measures: this model stops on
                    # tau.norm() < 1e-4, so by the time a frame ends the
                    # last step moved the pose less than that and a render
                    # taken one step earlier is all but identical. Early in
                    # a frame the steps are larger, but there the frame is
                    # nowhere near returning.
                    _pkg, _image, _depth, _opacity, _loss = _gr_last[0]
                else:
                    _pkg = render(
                        viewpoint,
                        self.gaussians,
                        self.pipeline_params,
                        self.background,
                        tracking_only=True,
                        tile_mask=_mask,
                        binning_kwargs=_bk,
                    )
                    _image, _depth, _opacity = (
                        _pkg["render"],
                        _pkg["depth"],
                        _pkg["opacity"],
                    )
                    # set_to_none=False is REQUIRED under capture. With
                    # set_to_none=True the .grad tensors are freed and
                    # reallocated every iteration, so the addresses the
                    # captured kernels were recorded against stop being the
                    # addresses autograd writes to.
                    pose_optimizer.zero_grad(set_to_none=False)
                    _loss = get_loss_tracking(
                        self.config, _image, _depth, _opacity, viewpoint
                    )
                    _loss.backward()
                    # BETWEEN backward() and the step: the next iteration's
                    # zero_grad clears .grad in place, so this is the only
                    # window in which there is anything to stash.
                    if self.grad_reuse.enabled:
                        _gr_stash[0] = stash_grads(_gr_pose_params)
                        _gr_last[0] = (_pkg, _image, _depth, _opacity, _loss)
                    if self.iter_trace.active:
                        # Before the step: update_pose zeroes the deltas and
                        # the next zero_grad clears .grad.
                        self.iter_trace.capture_tangent(
                            viewpoint.cam_trans_delta.grad,
                            viewpoint.cam_rot_delta.grad)
                with torch.no_grad():
                    # Pre-step snapshot of the deltas. They are always zero
                    # here - update_pose zeroes them at the end of every
                    # iteration, and a fresh Camera starts at zero - so this
                    # makes the recorded pose_delta come out as exactly
                    # ||tau||, the quantity the eager path EMAs. Recorded
                    # rather than assumed, because _prev_pose is not cleared by
                    # reset_for_frame.
                    self.es_signals.record_pre_step(
                        viewpoint.cam_rot_delta, viewpoint.cam_trans_delta
                    )
                    # THE GRADIENT IS ALREADY THE TANGENT 6-VECTOR here -
                    # cam_rot_delta and cam_trans_delta ARE the SE(3) tangent,
                    # so nothing has to be mapped across.
                    _gxi = None
                    if self.pose_pre is not None:
                        _gxi = torch.cat([
                            viewpoint.cam_rot_delta.grad.detach().flatten(),
                            viewpoint.cam_trans_delta.grad.detach().flatten(),
                        ])
                    _gxi_out[0] = _gxi
                    # Adam still steps: it also owns exposure_a / exposure_b,
                    # which tracking optimises and the preconditioner does not
                    # touch. Its POSE step is overwritten below when the
                    # preconditioner is driving - cheaper and far less fragile
                    # than splitting the optimiser in two.
                    pose_optimizer.step()
                    if _gxi is not None:
                        if _use_pre:
                            # A reused gradient is the same measurement
                            # twice, so under GRAD_REUSE_NOACC it drives a
                            # step without updating m or M. Measured WORSE
                            # on SplaTAM - freezing both moments makes the
                            # two steps identical, which doubles the
                            # effective step and took ATE 3.5 -> 10.8 - so
                            # it is off by default and kept only as an arm.
                            _dxi = self.pose_pre.step(
                                _gxi,
                                accumulate=not (_gr_reused[0]
                                                and self.grad_reuse.no_accum),
                                accumulate_m=not (_gr_reused[0]
                                                  and (self.grad_reuse.no_accum
                                                       or self.grad_reuse.freeze_m)))
                            viewpoint.cam_rot_delta.copy_(
                                _dxi[:3].view_as(viewpoint.cam_rot_delta))
                            viewpoint.cam_trans_delta.copy_(
                                _dxi[3:].view_as(viewpoint.cam_trans_delta))
                        else:
                            # ADAM PHASE OF A HANDOFF. Keep feeding the plateau
                            # tracker - step() is not being called, so _g_best
                            # and the stall counter would FREEZE at the handoff
                            # iteration and a frame that had not already
                            # stalled could never stop.
                            self.pose_pre.observe(_gxi)
                    # tau must be read HERE, after the step and before
                    # update_pose, because update_pose zeroes cam_rot_delta and
                    # cam_trans_delta IN PLACE before returning. Reading them
                    # afterwards always yields zeros - the bug that made every
                    # retune's pose diagnostic report 0.00e+00 across all 773
                    # probe samples.
                    _tau = torch.cat([
                        viewpoint.cam_rot_delta.detach().flatten(),
                        viewpoint.cam_trans_delta.detach().flatten(),
                    ])
                    _tau_norm = _tau.norm()
                    _conv = update_pose(
                        viewpoint, converged_threshold=self.native_conv_threshold)
                    if _gr_reused[0]:
                        # A REUSE ITERATION CANNOT REPORT CONVERGENCE.
                        #
                        # This model stops on tau.norm() < 1e-4, a pure
                        # STEP-MAGNITUDE test. A reuse iteration steps on a
                        # gradient that is one iteration old, so as the true
                        # gradient approaches zero the reused one does not,
                        # and the step never shrinks past the threshold -
                        # convergence becomes unreachable BY CONSTRUCTION.
                        #
                        # Measured before this guard: every frame ran to
                        # the 100-iteration cap with 0% spread, against 69.7
                        # it/frame and 69% spread without reuse. Tracking
                        # per FRAME went UP 21% even though per ITERATION it
                        # went down 15%, and Replica's ATE went 0.16 -> 1.20.
                        #
                        # SplaTAM was unaffected because its incumbent rule
                        # keys on loss improvement and pose change rather
                        # than on step norm. This is a property of the
                        # CRITERION, not of the model.
                        #
                        # Suppressing it here costs at most one iteration:
                        # the next RENDER iteration sees the true gradient
                        # and converges then.
                        _conv = (torch.zeros_like(_conv)
                                 if torch.is_tensor(_conv) else False)
                    self.es_signals.record_post_step(
                        _loss, _tau[:3], _tau[3:], extra=(_conv,)
                    )
                # _tau_norm and _conv are returned as DEVICE tensors. Reading
                # them is a sync, which is legal outside the capture but not
                # inside it, so the decision of whether to read them at all is
                # the caller's - see the two paths below. Under the graph these
                # are references into the graph's private pool, and replay
                # rewrites that same memory, so they keep reading current
                # values without re-running any Python.
                return _pkg, _loss, _conv, _tau_norm

            # RELEASE THE PREVIOUS ITERATION'S AUTOGRAD GRAPH BEFORE THE
            # CAPTURE ITERATION RUNS. Without this the capture dies in
            # backward() with:
            #
            #   RuntimeError: CUDA error: operation would make the legacy
            #   stream depend on a capturing blocking stream
            #
            # Autograd stamps every Node with the stream that was current at
            # FORWARD time, and the backward engine then runs each Node on its
            # stamped stream. AccumulateGrad is cached per leaf via a weak_ptr,
            # so as long as SOMETHING keeps the previous graph alive, the
            # accumulators built during the eager warmup - stamped with the
            # legacy default stream - survive and get reused inside the
            # capture. The engine then tries to make the legacy stream depend
            # on the capturing stream, which CUDA refuses.
            #
            # These names are what keeps it alive: they still hold iteration
            # N-1's tensors while iteration N captures. render_pkg matters as
            # much as the loss - it carries viewspace_points, which has
            # retain_grad() called on it, plus every render output. Drop them
            # and the weak_ptrs expire, so the accumulators are rebuilt inside
            # the capture on the capture stream.
            #
            # capture_error_mode does NOT help here. That setting controls
            # whether PyTorch POLICES other threads' default-stream use; this
            # is a genuine cross-stream dependency that CUDA rejects outright.
            #
            # Isolated on SplaTAM as fixB in profiling/repro_capture.py, and
            # carried in splatam.py as the bare `loss = losses = None` above
            # its own run() call.
            render_pkg = loss_tracking = converged = tau_norm = None
            image = depth = opacity = None

            if _adaptive_diag_enabled:
                # A full proposal cannot be validated until the NEXT render.
                # Split render/backward from update only for this experimental
                # arm; all fixed-D and disabled runs retain the original
                # combined iteration above byte-for-byte.
                render_pkg = render(
                    viewpoint,
                    self.gaussians,
                    self.pipeline_params,
                    self.background,
                    tracking_only=True,
                    tile_mask=_active_tile_mask,
                    binning_kwargs=_bin_kwargs,
                )
                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )
                pose_optimizer.zero_grad(set_to_none=False)
                loss_tracking = get_loss_tracking(
                    self.config, image, depth, opacity, viewpoint
                )
                loss_tracking.backward()

                _current_loss = None
                if not _adaptive_diag_active:
                    # This is the one unavoidable sync introduced by the
                    # online decision. It disappears permanently after the
                    # frame switches to diagonal mode.
                    _adaptive_sync_t0 = time.perf_counter()
                    _current_loss = float(loss_tracking.detach())
                    self._sync_time_total += (
                        time.perf_counter() - _adaptive_sync_t0
                    )
                _reject_full = False
                if _adaptive_pending_loss is not None:
                    _full_improved = adaptive_loss_improved(
                        _adaptive_pending_loss, _current_loss
                    )
                    (_adaptive_failure_count,
                     _reject_full) = adaptive_failure_streak(
                        _adaptive_failure_count,
                        _full_improved,
                        self.pose_pre.adaptive_diag_patience,
                    )
                if _reject_full:
                    with torch.no_grad():
                        # Restore the pose before the failed proposal, then
                        # replace that proposal with the diagonal step computed
                        # from the same retained gradient and metric.
                        viewpoint.update_RT(
                            _adaptive_pending_R, _adaptive_pending_T
                        )
                        _dxi = self.pose_pre.activate_adaptive_diagonal()
                        viewpoint.cam_rot_delta.copy_(
                            _dxi[:3].view_as(viewpoint.cam_rot_delta)
                        )
                        viewpoint.cam_trans_delta.copy_(
                            _dxi[3:].view_as(viewpoint.cam_trans_delta)
                        )
                        _tau = _dxi
                        tau_norm = _tau.norm()
                        converged = update_pose(
                            viewpoint, converged_threshold=self.native_conv_threshold)
                    _adaptive_diag_active = True
                    _adaptive_switch_iteration = tracking_itr
                    _adaptive_pending_loss = None
                    _adaptive_pending_R = None
                    _adaptive_pending_T = None
                    _adaptive_pending_objective = None
                    _adaptive_failure_count = 0

                    if self.es_signals.enabled:
                        # skip_iteration() is legal only with an empty pending
                        # ring. Native convergence uses this buffer even when
                        # windowed convergence is disabled, so the drain must
                        # follow the BUFFER rather than the P8 flag. The first
                        # calibrated-native run caught the old conditional at
                        # its first rejected proposal.
                        _phase_stop = _consume(self.es_signals.drain_phase())
                        _stop_now = _stop_now or _phase_stop
                    if self.windowed_stop.enabled:
                        # Acquisition rows cannot propose because these test
                        # configs require phase>=1. Open a fresh diagonal P8
                        # phase after consuming every valid acquisition row.
                        _window_phase_signature = (
                            _ctf_current_scale,
                            _active_tile_mask is not None,
                            "pre-diagonal",
                        )
                        _phase_label = (
                            f"scale={_window_phase_signature[0]},"
                            f"sparse={int(_window_phase_signature[1])},"
                            "update=pre-diagonal"
                        )
                        self.windowed_stop.start_phase(
                            tracking_itr + 1, _phase_label
                        )
                        if hasattr(
                            self.windowed_stop, "restart_phase_evidence"
                        ):
                            self.windowed_stop.restart_phase_evidence(
                                tracking_itr + 1
                            )
                        self.windowed_sweep.start_phase(
                            tracking_itr + 1, _phase_label
                        )
                    self.binning_capacity.note_iteration()
                    if self.es_signals.enabled:
                        # This render evaluated the rejected full pose; there
                        # is no truthful loss/step pair to record for it.
                        self.es_signals.skip_iteration()
                    self._tracking_time_total += (
                        time.perf_counter() - _iter_t0
                    )
                    self._tracking_iters_total += 1

                    if tracking_itr + 1 == self.tracking_itr_num:
                        # Keyframe logic below must see render products from the
                        # restored+diagonal pose, not the rejected trial pose.
                        with torch.no_grad():
                            render_pkg = render(
                                viewpoint,
                                self.gaussians,
                                self.pipeline_params,
                                self.background,
                                tracking_only=True,
                                tile_mask=_active_tile_mask,
                                binning_kwargs=_bin_kwargs,
                            )
                            image, depth, opacity = (
                                render_pkg["render"],
                                render_pkg["depth"],
                                render_pkg["opacity"],
                            )
                    continue

                if (not _adaptive_diag_active
                        and tracking_itr + 1 == self.tracking_itr_num):
                    # The final render has validated the previous full step,
                    # but a new proposal here could never be checked. Commit
                    # the validated pose instead of emitting an unvalidated
                    # cap-boundary step (and avoid an extra render).
                    self.binning_capacity.note_iteration()
                    if self.es_signals.enabled:
                        _stop_now = _stop_now or _consume(
                            self.es_signals.drain_phase()
                        )
                        self.es_signals.skip_iteration()
                    self._tracking_time_total += (
                        time.perf_counter() - _iter_t0
                    )
                    self._tracking_iters_total += 1
                    continue

                with torch.no_grad():
                    self.es_signals.record_pre_step(
                        viewpoint.cam_rot_delta,
                        viewpoint.cam_trans_delta,
                    )
                    _gxi = torch.cat([
                        viewpoint.cam_rot_delta.grad.detach().flatten(),
                        viewpoint.cam_trans_delta.grad.detach().flatten(),
                    ])
                    if not _adaptive_diag_active:
                        _adaptive_pending_loss = _current_loss
                        _adaptive_pending_R = viewpoint.R.detach().clone()
                        _adaptive_pending_T = viewpoint.T.detach().clone()
                        _adaptive_pending_objective = _objective_signature
                    pose_optimizer.step()
                    _dxi = self.pose_pre.step(_gxi)
                    viewpoint.cam_rot_delta.copy_(
                        _dxi[:3].view_as(viewpoint.cam_rot_delta)
                    )
                    viewpoint.cam_trans_delta.copy_(
                        _dxi[3:].view_as(viewpoint.cam_trans_delta)
                    )
                    _tau = torch.cat([
                        viewpoint.cam_rot_delta.detach().flatten(),
                        viewpoint.cam_trans_delta.detach().flatten(),
                    ])
                    tau_norm = _tau.norm()
                    converged = update_pose(
                        viewpoint, converged_threshold=self.native_conv_threshold)
                    self.es_signals.record_post_step(
                        loss_tracking,
                        _tau[:3],
                        _tau[3:],
                        extra=(converged,),
                    )
            else:
                render_pkg, loss_tracking, converged, tau_norm = self.iter_graph.run(
                    _tracking_iteration, bypass=_gr_reused[0]
                )
                if self.iter_trace.active:
                    self.iter_trace.note_iteration(
                        tracking_itr, loss_tracking, rot_to_quat(viewpoint.R),
                        viewpoint.T, reused=_gr_reused[0])
                # Adaptive-reuse trust signal, HERE and not in the closure: a
                # replayed iteration runs no Python. Render iterations only -
                # on a reuse iteration .grad was restored from the stash, so
                # comparing it to the last fresh gradient would read cos=1.
                # None (no preconditioner, or no gradient yet) counts as no
                # evidence, not distrust. A no-op unless adaptive is on.
                if self.grad_reuse.enabled and not _gr_reused[0]:
                    self.grad_reuse.note_fresh_grad(_gxi_out[0], tracking_itr)
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            _sync_t0 = time.perf_counter()
            if _use_pre:
                # REBUILD P. The eigendecomposition is not capturable and does
                # not belong inside the iteration, so it lives here - exactly
                # as in SplaTAM. Without this call P stays at the identity and
                # every step takes the no-metric fallback, which looks like a
                # working run and measures nothing.
                #
                # SKIPPED DURING THE ADAM PHASE: refactor() is also where the
                # step counter advances, so counting it on iterations that
                # never called step() dilutes every step statistic by the ratio
                # of the two - it read 0.02x Adam where the truth was 0.21x on
                # SplaTAM - and it rebuilds a metric nothing is using.
                self.pose_pre.refactor()
            self.binning_capacity.note_iteration()
            if self.es_signals.enabled:
                # Nothing is read back here. loss_tracking and converged stay on
                # the device; the drain below syncs once every `batch`
                # iterations instead of three times every iteration
                # (pose delta, loss, and the `if converged` bool).
                self.es_signals.note_iteration()
                _stop_now = _stop_now or _consume(
                    self.es_signals.drain_if_full()
                )
            elif self.stopper.enabled:
                # Eager path. The pose delta is read back here because
                # early-stop needs it on the host; with stopper.enabled=False
                # this block collapses to nothing, so a disabled run is a true
                # A/B baseline rather than "check() always returns False" while
                # still paying for the sync.
                #
                # tau is read from the value the iteration returned rather than
                # from viewpoint.cam_rot_delta, because update_pose now runs
                # inside the iteration and has already zeroed those in place.
                _raw_pose_delta_norm = tau_norm.item()
                # EMA-smooth before handing to early-stop: the raw per-step
                # delta is noisy enough that individual steps can look "still
                # moving" long after the pose has genuinely settled, which is
                # what was driving pose_tail_mass above the tuner's threshold
                # and forcing pose_eps to 0 every retune (loss-only stopping,
                # decoupled from actual pose convergence).
                _pose_delta_ema = (
                    _raw_pose_delta_norm if _pose_delta_ema is None else
                    _pose_ema_alpha * _raw_pose_delta_norm + (1 - _pose_ema_alpha) * _pose_delta_ema
                )
                pose_delta_norm = _pose_delta_ema
            # This block's wall time includes whatever host syncs remain, which
            # drain the queued forward/backward/step work - compared against
            # total tracking wall time, this bounds how much of the loop is
            # sync-serialized vs. able to pipeline across iterations.
            self._sync_time_total += time.perf_counter() - _sync_t0

            if self.use_gui and tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            self._tracking_time_total += time.perf_counter() - _iter_t0
            self._tracking_iters_total += 1
            # loss_tracking.item() is itself a host sync, and `if converged`
            # converts a device tensor to a Python bool, which is another -
            # both are skipped when disabled, so a stopper.enabled=False run is
            # a true A/B baseline against the enabled case, not just "check()
            # always returns False" while still paying for the syncs.
            _stop_reason = None
            if _force_dense:
                # The full-resolution iteration just ran. Commit and stop -
                # unconditionally, so a criterion that no longer fires cannot
                # leave the frame running at full resolution.
                _should_stop = True
                _stop_reason = _pending_stop_reason
            elif self.es_signals.enabled:
                # Every read already happened in _consume, at the drain. The
                # decision may be seen up to batch-1 iterations late; those
                # extra iterations still ran, but they only refine the pose
                # further, so the frame's committed pose cannot be worse.
                _should_stop = _stop_now
                if _should_stop:
                    if _window_stop_now and self.windowed_stop.active:
                        _stop_reason = "window"
                    else:
                        _stop_reason = (
                            "native" if _converged_now else "loss"
                        )
            else:
                _floor_open = (
                    self.pose_pre is None
                    or tracking_itr + 1 >= self.pre_hard_min
                )
                _native_stop = _floor_open and bool(converged)
                _loss_stop = False
                if (not _native_stop and _floor_open
                        and self.stopper.enabled):
                    _loss_stop = self.stopper.check(
                        tracking_itr, loss_tracking.item(), pose_delta_norm
                    )
                _should_stop = _native_stop or _loss_stop
                if _native_stop:
                    _stop_reason = "native"
                elif _loss_stop:
                    _stop_reason = "loss"
            # PLATEAU STOPPING: a patience on the BEST |g| this frame, not a
            # threshold on the current one. |g|/|g_0| is non-monotone and
            # plateaus, so a threshold either fires on a noise dip or waits far
            # too long; and stalled() is gated on the frame having actually
            # beaten its own starting gradient, so a DIVERGING frame - which
            # also stops improving - runs its full budget instead of quitting
            # at the floor. Both were measured on SplaTAM.
            #
            # Checked every pre_stop_every iterations: float() on a device
            # tensor is a host sync, and one per iteration is a bad trade.
            if (not _should_stop and self.pose_pre is not None
                    and (not self.windowed_stop.active
                         or self.windowed_plateau_fallback)
                    and self.pre_stop_rel > 0.0
                    and self.pre_stop_mode == "best"
                    and tracking_itr >= self.pre_stop_min
                    and tracking_itr + 1 >= self.pre_hard_min
                    and tracking_itr % self.pre_stop_every == 0):
                if float(self.pose_pre.stalled()) >= self.pre_stop_patience:
                    _should_stop = True
                    _stop_reason = "plateau"
            if _should_stop and self.pre_final_dense and not _force_dense:
                # ONE FULL-RESOLUTION OPTIMISATION ITERATION before committing.
                #
                # NOT THE SAME THING AS THE CORRECTIVE RENDER BELOW, and they
                # must not be merged. That one is torch.no_grad() and exists to
                # give median_depth and n_touched an unmasked view for keyframe
                # selection - it explicitly does not feed the pose optimiser.
                # This one is a real gradient step, so the pose itself gets one
                # look at the full image before the frame commits. On SplaTAM
                # its absence cost ATE 7.96 against 4.04 for 1.5 extra
                # iterations per frame.
                _pending_stop_reason = _stop_reason
                _force_dense = True
                _should_stop = False
            if _should_stop:
                _frame_stop_reason = _stop_reason or "cap"
                _mid_coarse = _ctf_enabled and _ctf_current_scale != 1
                if _mid_coarse:
                    # Firing while still downsampled: restore full
                    # resolution before the corrective render below reads
                    # viewpoint state. Structurally shouldn't be reachable
                    # at min_iters=50 (coarse window closes at iter 30),
                    # but defends against it if that floor is ever lowered.
                    viewpoint.original_image = _orig_image
                    viewpoint.depth = _orig_depth
                    viewpoint.grad_mask = _orig_grad_mask
                    viewpoint.image_height = _orig_h
                    viewpoint.image_width = _orig_w
                    _ctf_current_scale = 1
                if _active_tile_mask is not None or _mid_coarse:
                    # Firing while still in the sparse phase and/or still
                    # downsampled: median_depth and n_touched (is_keyframe's
                    # visibility input) must never come from a partially-
                    # masked or reduced-resolution render - masked tiles
                    # report depth=0/opacity=0/n_touched=0, and a
                    # downsampled render biases the same values less
                    # drastically but still incorrectly. One extra full-
                    # pixel, full-resolution render here (no grad needed,
                    # doesn't feed the pose optimizer) fixes this regardless
                    # of when early-stop decides to fire, instead of
                    # disabling early-stop whenever these features are on
                    # (what the reference implementations this was ported
                    # from do).
                    self._corrective_render_count += 1
                    with torch.no_grad():
                        render_pkg = render(
                            viewpoint,
                            self.gaussians,
                            self.pipeline_params,
                            self.background,
                            tracking_only=True,
                        )
                        depth, opacity = render_pkg["depth"], render_pkg["opacity"]
                break

        # Closes "monogs_tracking" - see the push above the loop. Placed after
        # the loop's own break paths so it is closed exactly once per frame
        # regardless of how tracking ended.
        torch.cuda.nvtx.range_pop()
        if self.windowed_stop.enabled:
            # A 100-iteration shadow run with batch=8 leaves four rows in the
            # ring. Drain them for evidence only; tracking has already ended,
            # so a proposal here cannot change the committed work or reason.
            for _r in self.es_signals.drain_remainder():
                _step = np.asarray(
                    list(_r["rot"]) + list(_r["tran"]), dtype=np.float64
                )
                self.windowed_stop.observe(_r["iter"], _r["loss"], _step)
                self.windowed_sweep.observe(
                    _r["iter"], _r["loss"], _step
                )
            self.windowed_stop.end_frame()
            self.windowed_sweep.end_frame()
        if self._tail_diag:
            # drain_remainder above runs only when windowed convergence is on.
            # With WCONV=0 the last rows of every frame (100 % batch 8 = 4)
            # are never consumed, and those are exactly the iterations this
            # diagnostic is about. Drain them here.
            if not self.windowed_stop.enabled:
                _consume(self.es_signals.drain_remainder())
            if (_tail_first is not None and _tail_best is not None
                    and _tail_ref_loss is not None):
                _den = max(abs(_tail_ref_loss), 1e-12)
                _bden = max(abs(_tail_best), 1e-12)
                self._tail_rows.append((
                    _tail_best_it,
                    (_tail_ref_loss - _tail_best) / _den,
                    (_tail_final - _tail_best) / _bden,
                    (_tail_first - _tail_ref_loss) / max(abs(_tail_first), 1e-12),
                ))
        self._tracking_stop_counts[_frame_stop_reason] += 1
        # PER-FRAME ITERATION COUNT, independent of whether reuse is on - the
        # aggregate "Tracking iterations/frame: X.X" line only prints once at
        # the very end, which hides how much any individual frame varies
        # (cap vs. an early native/window/plateau stop). Unconditional: this
        # is exactly what GRAD_REUSE=0 baseline runs need to be comparable
        # against the reuse runs' per-frame numbers.
        Log(f"[Tracking] frame {cur_frame_idx}: {tracking_itr + 1} iters "
            f"(stop={_frame_stop_reason})")
        self.grad_reuse.note_frame_stop(
            tracking_itr + 1,
            _frame_stop_reason,
        )
        if _sig_trace:
            self._flush_signal_trace(_sig_trace, _frame_stop_reason)
        if _adaptive_calibration_frame:
            _learned_diag_after = (
                self.pose_pre.finish_adaptive_calibration_frame(
                    _adaptive_switch_iteration,
                    tracking_itr + 1,
                )
            )
            if _learned_diag_after is not None:
                Log(
                    "[Preconditioner] adaptive calibration complete: "
                    f"fixed restart_at=diag_after={_learned_diag_after}; "
                    "future frames use the combined iteration without the "
                    "adaptive loss readback"
                )

        # Always restore full-resolution viewpoint after tracking (safety
        # net for any code path that reads it afterward), even though the
        # loop's own scale-switching plus the mid-coarse break handling
        # above should already guarantee this.
        if _ctf_enabled and _ctf_current_scale != 1:
            viewpoint.original_image = _orig_image
            viewpoint.depth = _orig_depth
            viewpoint.grad_mask = _orig_grad_mask
            viewpoint.image_height = _orig_h
            viewpoint.image_width = _orig_w

        # One host read per frame instead of ~100: an overflow means
        # duplicateWithKeys dropped instances and this frame's render is wrong,
        # so it is reported loudly rather than absorbed.
        self.binning_capacity.end_frame()

        if self.iter_graph.enabled:
            # ANYTHING THAT OUTLIVES THE ITERATION MUST BE CLONED OUT.
            #
            # render_pkg's tensors were produced inside the capture, so they
            # live in the graph's private memory pool and every replay - this
            # frame's and, after the next reset_for_frame, the NEXT frame's -
            # rewrites that same memory in place. The caller stores these:
            # visibility_filter goes into self.occ_aware_visibility, n_touched
            # feeds is_keyframe, and the whole packet is handed to the backend.
            # Left as views into the pool they would silently start reporting
            # some later frame's values.
            #
            # This is the failure that produced 591/591 frames captured at ATE
            # 156.40cm in SplaTAM - it looks exactly like a success.
            render_pkg = {
                k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in render_pkg.items()
            }
            depth, opacity = render_pkg["depth"], render_pkg["opacity"]

        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        # KF_TRANS / KF_MIN_TRANS / KF_OVERLAP override the YAML.
        #
        # WHY THIS IS WORTH A KNOB. The graph-on run produced 126 keyframes
        # against graph-off's 75 and reached ATE 1.607 vs 1.867 - but the graph
        # is a 35% in-loop REGRESSION, so it cannot be improving tracking. The
        # likely chain is: slower frontend -> different async pacing -> 68% more
        # keyframes -> more mapping -> better map. If that holds, keyframe count
        # is a strong ATE lever on MonoGS that is currently an EMERGENT
        # PROPERTY OF TIMING rather than a tuned parameter, and it should be
        # set deliberately instead of obtained by making the frontend slow.
        #
        # The criterion is
        #   (overlap < kf_overlap and dist > kf_min_translation*d)
        #     or dist > kf_translation*d
        # so lowering kf_translation toward kf_min_translation collapses the OR
        # to the permissive branch - the most direct way to get more keyframes.
        kf_translation = float(os.environ.get(
            "KF_TRANS", self.config["Training"]["kf_translation"]))
        kf_min_translation = float(os.environ.get(
            "KF_MIN_TRANS", self.config["Training"]["kf_min_translation"]))
        kf_overlap = float(os.environ.get(
            "KF_OVERLAP", self.config["Training"]["kf_overlap"]))

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            # empty_cache() forces a full CUDA sync and releases cached
            # allocator memory back to the driver - a known-expensive call
            # if it fires often in a hot loop. This is one of two call
            # sites (the other, inside kf_logic, is already covered by
            # that timer) - this one runs from the early-exit branch that
            # ~40% of frames take, entirely before any per-frame timer
            # starts, so it's a real candidate for the still-unaccounted
            # wall time.
            _t0 = time.perf_counter()
            torch.cuda.empty_cache()
            self._empty_cache_time_total += time.perf_counter() - _t0
            self._empty_cache_calls += 1

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self._mlsys_track_frames > 0:
                        Log("Tracking Total time", self._mlsys_track_time, tag="Eval")
                        Log("Tracking FPS", self._mlsys_track_frames / self._mlsys_track_time, tag="Eval")
                        Log(f"Tracking ms/frame: {1000.0 * self._mlsys_track_time / self._mlsys_track_frames:.4f}", tag="Eval")
                        Log(f"Tracking ms/iter: "
                            f"{1000.0 * self._mlsys_track_time / max(self._mlsys_track_iters, 1):.4f}", tag="Eval")
                        Log(f"Tracking iters/frame: {self._mlsys_track_iters / self._mlsys_track_frames:.4f}", tag="Eval")
                    # VERBOSE_DIAG=1 restores the early-stop and signal-
                    # batching-detail lines. Default off, matching SplaTAM's
                    # same trim - kept: gradient reuse, the CUDA graph
                    # capture, binning capacity (ties into graph capture),
                    # and the windowed-convergence (incumbent) stopper.
                    if os.environ.get("VERBOSE_DIAG", "0") not in ("0", "", "false", "False"):
                        Log(self.stopper.summary())
                    Log(self.iter_graph.summary())
                    Log(self.grad_reuse.summary())
                    Log(self.binning_capacity.summary())
                    if os.environ.get("VERBOSE_DIAG", "0") not in ("0", "", "false", "False"):
                        Log(self.es_signals.summary())
                    Log(self.windowed_stop.summary(self.tracking_itr_num))
                    for _line in self.windowed_sweep.summary_lines(
                        self.tracking_itr_num
                    ):
                        Log(_line)
                    # READ THIS BEFORE BELIEVING ANY PREFETCH TIMING. A
                    # prefetcher that misses every frame still runs and still
                    # buys nothing; "consumer blocked" near zero with a high hit
                    # rate is what says the load was actually hidden.
                    if hasattr(self.dataset, "summary"):
                        Log(self.dataset.summary())
                    if self._tracking_iters_total > 0:
                        _sync_frac = self._sync_time_total / self._tracking_time_total
                        Log(
                            f"Tracking sync diagnostic: {self._tracking_iters_total} iters, "
                            f"{1000 * self._tracking_time_total / self._tracking_iters_total:.2f} ms/iter avg, "
                            f"{1000 * self._sync_time_total / self._tracking_iters_total:.2f} ms/iter in "
                            f"the .item()-sync block ({100 * _sync_frac:.1f}% of tracking wall time)"
                        )
                    if self._frame_loop_count > 0:
                        # The "total" here is only accumulated inside the
                        # `frontend_queue.empty()` branch, while every
                        # component below accumulates on every pass. With an
                        # ASYNC backend the loop takes the other branch
                        # constantly to service backend messages, so the total
                        # covers a fraction of the run while the parts cover
                        # all of it - the first async run reported
                        # "133 frames (total 61.5s)" with "tracking=208.1s"
                        # inside it, which is arithmetically impossible.
                        #
                        # Say so rather than printing a plausible-looking line.
                        # This is inline-mode-only bookkeeping; in async mode
                        # the components are still individually correct, it is
                        # only their attribution to a whole that is not.
                        _parts = (self._load_time_total + self._tracking_time_total
                                  + self._gui_time_total + self._kf_logic_time_total
                                  + self._eval_time_total
                                  + self._throttle_sleep_time_total)
                        if _parts > self._frame_loop_time_total:
                            Log(
                                f"Per-frame breakdown: the frame-loop total "
                                f"({self._frame_loop_time_total:.1f}s over "
                                f"{self._frame_loop_count} counted passes) is "
                                f"SMALLER than the sum of its parts "
                                f"({_parts:.1f}s), so it is not a valid total - "
                                f"it only accumulates on passes that take the "
                                f"tracking branch, which an async backend "
                                f"leaves in the minority. The components below "
                                f"are each still correct; use 'Total time' for "
                                f"the run, and compare tracking against "
                                f"tracking."
                            )
                        Log(
                            f"Per-frame breakdown over {self._frame_loop_count} counted passes "
                            f"(frame-loop total {self._frame_loop_time_total:.1f}s): "
                            f"load={self._load_time_total:.1f}s "
                            f"tracking={self._tracking_time_total:.1f}s "
                            f"gui={self._gui_time_total:.1f}s "
                            f"kf_logic={self._kf_logic_time_total:.1f}s "
                            f"eval={self._eval_time_total:.1f}s "
                            f"throttle_sleep={self._throttle_sleep_time_total:.1f}s"
                        )
                        Log(
                            f"Backend-message handling: {self._backend_msg_time_total:.1f}s "
                            f"over {sum(self._backend_msg_counts.values())} messages "
                            f"{self._backend_msg_counts}"
                        )
                        Log(
                            f"cuda.empty_cache(): {self._empty_cache_time_total:.1f}s "
                            f"over {self._empty_cache_calls} calls "
                            f"({1000 * self._empty_cache_time_total / self._empty_cache_calls:.1f} ms/call avg)"
                            if self._empty_cache_calls > 0 else
                            "cuda.empty_cache(): never called"
                        )
                        Log(
                            f"Startup/init wait: requested_init={self._init_wait_time_total:.1f}s "
                            f"single_thread={self._single_thread_wait_time_total:.1f}s "
                            f"not_initialized={self._not_initialized_wait_time_total:.1f}s"
                        )
                        Log(
                            f"Sparse tracking: {self._sparse_iters_total}/{self._tracking_iters_total} "
                            f"iters masked ({100 * self._sparse_iters_total / max(self._tracking_iters_total, 1):.1f}%)"
                        )
                        Log(
                            f"Coarse-to-fine: {self._ctf_iters_total}/{self._tracking_iters_total} "
                            f"iters downsampled ({100 * self._ctf_iters_total / max(self._tracking_iters_total, 1):.1f}%)"
                        )
                        Log(
                            f"{self._corrective_render_count} corrective full-resolution/unmasked renders "
                            f"(converged or early-stop fired while still sparse and/or downsampled)"
                        )
                        Log(self.pose_pre.summary()
                            if self.pose_pre is not None
                            else "Pose preconditioner: disabled")
                        if self.pose_pre is not None:
                            _sp = self.pose_pre.step_profile()
                            if _sp:
                                Log(_sp)
                        # NOT gated on the preconditioner. These two lines
                        # were, and the denominator was pose_pre.frames, so an
                        # Adam control printed neither - which made an
                        # iso-iteration claim about a precond/Adam pair
                        # unverifiable from its own logs. _tracking_stop_counts
                        # is incremented once per tracked frame regardless of
                        # the update rule, so its sum is the frame count both
                        # arms can report.
                        _sc = self._tracking_stop_counts
                        _frames = sum(_sc.values())
                        if _frames == 0 and self.pose_pre is not None:
                            _frames = self.pose_pre.frames
                        Log(f"Tracking iterations/frame: "
                            f"{self._tracking_iters_total / max(_frames, 1):.1f}"
                            f"  (cap {self.tracking_itr_num}, {_frames} frames)")
                        Log("Tracking stop reasons: "
                            f"native={_sc['native']} loss={_sc['loss']} "
                            f"plateau={_sc['plateau']} "
                            f"window={_sc['window']} cap={_sc['cap']} "
                            f"(hard floor {self.pre_hard_min})")
                        if self._tail_diag and self._tail_rows:
                            _a = np.asarray(self._tail_rows, dtype=np.float64)
                            _bi = _a[:, 0]
                            _ref = self._tail_ref
                            _q = np.percentile(_bi, [10, 50, 90])
                            Log(
                                "TAIL DIAGNOSTIC over "
                                f"{len(_a):.0f} frames (ref iteration {_ref}, "
                                f"cap {self.tracking_itr_num}):")
                            Log(f"  best-loss iteration p10/med/p90 = "
                                f"{_q[0]:.0f}/{_q[1]:.0f}/{_q[2]:.0f}; "
                                f"{100 * float((_bi <= _ref).mean()):.1f}% of "
                                f"frames peaked at or before it{_ref}")
                            Log(f"  loss fall it0->it{_ref}: med "
                                f"{100 * float(np.median(_a[:, 3])):.2f}%   "
                                f"loss fall it{_ref}->best: med "
                                f"{100 * float(np.median(_a[:, 1])):.2f}% "
                                f"p90 {100 * float(np.percentile(_a[:, 1], 90)):.2f}%")
                            Log(f"  COMMIT PENALTY (final vs best loss): med "
                                f"{100 * float(np.median(_a[:, 2])):.2f}% "
                                f"p90 {100 * float(np.percentile(_a[:, 2], 90)):.2f}% "
                                f"max {100 * float(_a[:, 2].max()):.2f}%  "
                                "- MonoGS commits the LAST pose, not the best")
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    _t0 = time.perf_counter()
                    time.sleep(0.01)
                    self._init_wait_time_total += time.perf_counter() - _t0
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    _t0 = time.perf_counter()
                    time.sleep(0.01)
                    self._single_thread_wait_time_total += time.perf_counter() - _t0
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    _t0 = time.perf_counter()
                    time.sleep(0.01)
                    self._not_initialized_wait_time_total += time.perf_counter() - _t0
                    continue

                _frame_t0 = time.perf_counter()
                _load_t0 = time.perf_counter()
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix,
                    device=self.device,
                )
                viewpoint.compute_grad_mask(self.config)
                self._load_time_total += time.perf_counter() - _load_t0

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                # mlsys CSV: wall time (synchronised) and iteration count of each tracking() call,
                # same definition as the other MonoGS repos' Eval: Tracking ms/frame lines.
                _mlsys_t0 = time.time()
                _mlsys_it0 = self._tracking_iters_total
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                torch.cuda.synchronize()
                self._mlsys_track_time += time.time() - _mlsys_t0
                self._mlsys_track_frames += 1
                self._mlsys_track_iters += self._tracking_iters_total - _mlsys_it0
                if self.iter_trace.active:
                    self.iter_trace.end_frame(rot_to_quat(viewpoint.R),
                                              viewpoint.T)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                # clone_obj(self.gaussians) deep-copies the entire Gaussian
                # model (every frame, unconditionally) to hand to the GUI
                # process - with use_gui=False, q_main2vis is a FakeQueue
                # whose .put() immediately discards it (`del arg`), but the
                # clone itself is a function argument, evaluated whether or
                # not anything reads the queue. Skip building the packet at
                # all when there's no GUI to consume it.
                _gui_t0 = time.perf_counter()
                if self.use_gui:
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            current_frame=viewpoint,
                            keyframes=keyframes,
                            kf_window=current_window_dict,
                        )
                    )
                self._gui_time_total += time.perf_counter() - _gui_t0

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                _kf_t0 = time.perf_counter()
                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                if self.inline_backend and self.backend is not None:
                    # Substitute for BackEnd.run()'s continuous background-
                    # mapping loop, which never executes when
                    # use_inline_backend is on - see idle_mapping() in
                    # slam_backend.py. Ported from RTGS.
                    self._idle_frame_counter += 1
                    if self._idle_frame_counter % self.idle_mapping_interval == 0:
                        self.backend.idle_mapping()
                cur_frame_idx += 1
                self._kf_logic_time_total += time.perf_counter() - _kf_t0

                _eval_t0 = time.perf_counter()
                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                self._eval_time_total += time.perf_counter() - _eval_t0
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    _sleep_t0 = time.perf_counter()
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
                    self._throttle_sleep_time_total += time.perf_counter() - _sleep_t0
                self._frame_loop_time_total += time.perf_counter() - _frame_t0
                self._frame_loop_count += 1
            else:
                _msg_t0 = time.perf_counter()
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
                self._backend_msg_time_total += time.perf_counter() - _msg_t0
                self._backend_msg_counts[data[0]] = self._backend_msg_counts.get(data[0], 0) + 1
