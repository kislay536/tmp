""" This module includes the Mapper class, which is responsible scene mapping: Paper Section 3.4  """
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.spatial.transform import Rotation as R

from src.entities.arguments import OptimizationParams
from src.entities.losses import l1_loss
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.datasets import BaseDataset
from src.entities.visual_odometer import VisualOdometer
from src.utils.gaussian_model_utils import build_rotation
from src.utils.tracker_utils import (compute_camera_opt_params,
                                     extrapolate_poses, multiply_quaternions,
                                     transformation_to_quaternion)
from src.utils.utils import (get_render_settings, np2torch,
                             render_gaussian_model, torch2np)
from utils.pixel_sample import build_tile_mask, kept_pixel_fraction
from utils.grad_reuse import (GradReuse, RenderClock, stash_grads, restore_grads,
                              config_from_env as grad_reuse_env)

import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from utils.binning_capacity import BinningCapacity
from utils.early_stop import EarlyStop
from utils.es_signals import ESSignalBuffer
from utils.tracking_iteration_graph import TrackingIterationGraph
from utils.grad_staleness import GradStalenessProbe
from utils.iter_trace import IterTrace
from utils.pose_preconditioner import (PosePreconditioner,
                                       quat_trans_grad_to_tangent,
                                       apply_tangent_step,
                                       mat_to_quat_capturable,
                                       tangent_of_pose_delta,
                                       tangent_of_pose_delta_batch,
                                       se3_adjoint,
                                       adaptive_loss_improved,
                                       adaptive_failure_streak)
from utils.online_lr_tuner import PoseErrDelta
from utils.windowed_convergence import (
    WindowedConvergenceSweep,
    _pose_distance_numpy,
    make_windowed_convergence,
)
from utils.closed_loop_stop_guard import AuditedStopGuard


class Tracker(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:
        """ Initializes the Tracker with a given configuration, dataset, and logger.
        Args:
            config: Configuration dictionary specifying hyperparameters and operational settings.
            dataset: The dataset object providing access to the sequence of frames.
            logger: Logger object for logging the tracking process.
        """
        self.dataset = dataset
        self.logger = logger
        self.config = config
        self.filter_alpha = self.config["filter_alpha"]
        self.filter_outlier_depth = self.config["filter_outlier_depth"]
        self.alpha_thre = self.config["alpha_thre"]
        self.soft_alpha = self.config["soft_alpha"]
        self.mask_invalid_depth_in_color_loss = self.config["mask_invalid_depth"]
        self.w_color_loss = self.config["w_color_loss"]
        self.transform = torchvision.transforms.ToTensor()
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.frame_depth_loss = []
        self.frame_color_loss = []
        self.odometry_type = self.config["odometry_type"]
        self.help_camera_initialization = self.config["help_camera_initialization"]
        self.init_err_ratio = self.config["init_err_ratio"]
        self.odometer = VisualOdometer(self.dataset.intrinsics, self.config["odometer_method"])

        # A fixed binning capacity is the PREREQUISITE for the iteration graph,
        # not an independent optimisation - see the preflight below.
        self.binning_capacity = BinningCapacity(config.get("binning_capacity", {}))
        # TILE-LEVEL SPARSE SAMPLING. Ported to this model; it had none.
        #
        # ALWAYS-SPARSE IS FORCED, whatever the config says. is_sparse_phase
        # divides by the iteration CAP, so a middle window on a frame that
        # stops early ends inside the sparse phase and never reaches its
        # closing dense iterations. Rather than carry that trap, the window is
        # pinned to [0.0, 1.0) and the stop path grants one dense iteration
        # before committing. It is also the only shape the iteration graph
        # accepts, since a capture needs the mask constant within a frame.
        self.pixel_sample_cfg = dict(config.get("pixel_sample", {}))
        if "PIXEL_SAMPLE" in os.environ:
            self.pixel_sample_cfg["enabled"] = os.environ["PIXEL_SAMPLE"] not in (
                "0", "", "false", "False")
        if "PS_RATIO" in os.environ:
            self.pixel_sample_cfg["sample_ratio"] = float(os.environ["PS_RATIO"])
        # PS_GRADFRAC: what fraction of the KEPT tiles is chosen by image
        # gradient rather than uniformly at random.
        #
        # This is not a cosmetic knob. Gradient-ranked selection keeps the
        # sharpest tiles and preferentially drops the smooth ones - but the
        # smooth regions are what DAMP the loss surface, since they change
        # slowly with pose. Stripping them leaves a rougher objective than the
        # dense one, and on GSLAM/fr1 that put tracking into a limit cycle: at
        # frame 417 the loss swung 40% between iterations while cam_trans_err
        # moved under 1%, so no loss-based convergence rule could ever fire,
        # and the frame ended 27% worse than dense (0.0519 vs 0.0408).
        #
        # 0.0 is uniform tile sampling, which is an UNBIASED subsample of the
        # dense loss. 1.0 is purely gradient-ranked, the most biased and the
        # sharpest. The default stays 0.5 so no existing run changes.
        if "PS_GRADFRAC" in os.environ:
            self.pixel_sample_cfg["gradient_frac"] = float(
                os.environ["PS_GRADFRAC"])
        self.pixel_sample_cfg["full_start_ratio"] = 0.0
        self.pixel_sample_cfg["full_end_ratio"] = 1.0
        self.pixel_sample_cfg.setdefault("sample_ratio", 0.75)
        self.pixel_sample_cfg.setdefault("gradient_frac", 0.5)
        self.pixel_sample_cfg.setdefault("border_pixels", 10)
        self._ps_frames = 0
        self._ps_iters_masked = 0
        self._ps_dense_commits = 0
        print(f"[PixelSample] constructed (enabled="
              f"{self.pixel_sample_cfg.get('enabled', False)}, "
              f"sample_ratio={self.pixel_sample_cfg['sample_ratio']}, "
              f"window=[0.0, 1.0) always-sparse, forced dense before commit)",
              flush=True)
        _ig = dict(config.get("iteration_graph", {}))
        # ITERGRAPH=0 turns the capture off without editing the YAML. The
        # configs carrying the ladder have it ON, and the preconditioner
        # refuses to run with it - so testing the preconditioner on top of the
        # ladder needs a way to drop just this one piece.
        if "ITERGRAPH" in os.environ:
            _ig["enabled"] = os.environ["ITERGRAPH"] not in ("0", "", "false", "False")
        if "ITERGRAPH_WARMUP" in os.environ:
            _ig["warmup_iters"] = int(os.environ["ITERGRAPH_WARMUP"])
        self.iter_graph = TrackingIterationGraph(_ig)
        # TRACKING_ONLY=0 RESTORES THE STOCK BACKWARD.
        #
        # tracking_only skips gradients the tracking loss provably does not
        # consume - this branch's own optimisation, and the record marks it
        # "test unconfirmed". It is the only backward-altering flag GSLAM's
        # tracking render passes, so it is the cheapest suspect to clear while
        # investigating why this port sits ~2x above the paper's ATE on BOTH
        # TUM (2.6cm) and Replica.
        self.tracking_only_backward = (
            os.environ.get("TRACKING_ONLY", "1") not in ("0", "", "false", "False"))
        if not self.tracking_only_backward:
            print("[Tracker] TRACKING_ONLY=0: stock backward (all gradients)",
                  flush=True)
        # TRACK_ITERS PINS THE TRACKING BUDGET FOR AN ISO-ITERATION A/B.
        #
        # Comparing two update rules by ATE is only meaningful if both spent
        # the same number of iterations. Every GSLAM comparison so far has
        # confounded the two: the preconditioner arm ran 116 iters/frame and
        # the baseline 156, so "better ATE" and "fewer iterations" could not be
        # separated, and neither could "worse ms/iter" from "different work".
        #
        # This also has to defeat the adaptive doubling below (line ~402): it
        # fires on a high initial loss, which DIVERGES between arms as their
        # trajectories separate, so the budgets would drift apart mid-run
        # precisely on the hard frames that decide the ATE.
        self.fixed_iters = int(os.environ.get("TRACK_ITERS", "0") or 0)
        if self.fixed_iters > 0:
            print(f"[Tracker] TRACK_ITERS={self.fixed_iters}: budget PINNED, "
                  f"adaptive doubling disabled (iso-iteration A/B mode)",
                  flush=True)
        # Ported from exp/adaptive_mapping, which already carried this wiring -
        # same EarlyStop class, same reset/check contract, same pose-delta
        # definition (||pose_after - pose_before|| over the optimizer step).
        # What changes here is only WHERE the two signals are read: under
        # es_signals they arrive from the batched drain instead of a per-
        # iteration .item(), so the decisions are identical and only the sync
        # frequency differs.
        _es = dict(config.get("early_stop", {}))
        # ES=0/1 without editing the YAML, same name SplaTAM and MonoGS use.
        #
        # WHY IT MATTERS FOR THE PRECONDITIONER ARM. The ladder configs enable
        # early_stop, so its loss criterion and the plateau criterion are both
        # armed and whichever fires first wins - which makes a result on the
        # new criterion uninterpretable. The first Gaussian-SLAM preconditioner
        # run reached 39.1 iters/frame of a 200 budget with ES DISABLED, so the
        # plateau rule is doing real work here and deserves to be measured on
        # its own.
        if "ES" in os.environ:
            _es["enabled"] = os.environ["ES"] not in ("0", "", "false", "False")
        self.stopper = EarlyStop(_es)

        # FULL-MATRIX POSE PRECONDITIONER, ported from SplaTAM.
        #
        # THIS IS THE HARDER OF THE TWO PORTS. Gaussian-SLAM optimises
        # opt_cam_rot (a 4-element quaternion) and opt_cam_trans - the same
        # 7-parameter layout SplaTAM uses, NOT MonoGS's SE(3) tangent. So the
        # gradient has to be mapped across with quat_trans_grad_to_tangent and
        # the step applied as T <- exp(dxi) T, exactly as in SplaTAM. Three of
        # the four fixes that took there existed only because 7 parameters
        # describe a 6-dimensional object; the tangent path avoids all of them.
        #
        # THE TUM RESULT NEEDS ALL THE PIECES TOGETHER - preconditioner with
        # plateau stopping ALONE was 1/3 on fr1_desk, and 6/6 with the handoff.
        # final_dense is inert here: Gaussian-SLAM has no tile masking, so
        # there is no sparse phase for an early stop to cut short.
        _pre = dict(config.get("preconditioner", {}))
        _envmap = {
            "PRECOND": ("enabled", lambda v: v not in ("0", "", "false", "False")),
            "AUTO_LR": ("auto_lr", lambda v: v not in ("0", "", "false", "False")),
            "PRE_LR": ("lr", float), "PRE_TAU": ("tau", float),
            # BETA2 IS A HORIZON IN ITERATIONS, SO IT IS BUDGET-RELATIVE.
            # M remembers ~1/(1-beta2) samples for its 21 free parameters. The
            # shipped 0.95 gives 20, which is 10% of a 200-iteration GSLAM
            # frame but the WHOLE of a 20-iteration one - so an iso-iteration
            # A/B at a stressed budget changes the estimator's sample count
            # even though nothing about the rule changed. There was no hook to
            # hold it fixed or to rescale it; now there is. Default unchanged.
            "PRE_BETA1": ("beta1", float), "PRE_BETA2": ("beta2", float),
            "PRE_SHRINK": ("shrink", float), "PRE_REFACTOR": ("refactor_every", int),
            "CARRY": ("carry", lambda v: v not in ("0", "", "false", "False")),
            "STOP_REL": ("stop_rel_grad", float), "STOP_MODE": ("stop_mode", str),
            "STOP_PATIENCE": ("stop_patience", int),
            "STOP_IMPROVE": ("stop_improve", float),
            "STOP_MIN": ("stop_min_iters", int),
            "STOP_EVERY": ("stop_check_every", int),
            "PRE_HANDOFF": ("handoff", int),
            # PORTED FROM SplaTAM, where they took the handoff-free arm from
            # 2/4 to 14/16 over sixteen full-length runs.
            #
            # PRE_MAX_STEP=1.0 is NOT swept: |M^{-1/2} m|^2 = m^T M^-1 m <= n,
            # so ref = lr*sqrt(n) is the bound the estimator satisfies by
            # construction. The shipped default sits 10x above it - and the
            # record's unexplained "max |d| = 2.45e-01 EXACTLY" on this model
            # is that default (10 * 0.01 * sqrt(6)) being saturated.
            #
            # PRE_RESTART is an ITERATION COUNT and does not port across
            # budgets unchanged. GSLAM runs ~116-156 iters/frame against
            # SplaTAM's ~30, so 20 is a much earlier point in the frame here.
            "PRE_MAX_STEP": ("max_step_mult", float),
            "PRE_RESTART": ("restart_at", int),
            "PRE_RESTART_M": ("restart_m", str),
            "PRE_DIAG_AFTER": ("diag_after", int),
            # Optional per-coordinate rates for the diagonal tail.  The
            # acquisition phase keeps PRE_LR; either omitted tail rate falls
            # back to the scene config's native translation/rotation rate.
            "PRE_DIAG_TRANS_LR": ("diag_trans_lr", float),
            "PRE_DIAG_ROT_LR": ("diag_rot_lr", float),
            # THE CALIBRATED ALTERNATIVE TO A HAND-SET diag_after, ported from
            # MonoGS where it is the production arm. Instead of choosing an
            # iteration, it detects the one at which the full 6x6 stops
            # improving the frame's own loss: every full step is a proposal,
            # the next render scores it, and after `patience` consecutive
            # failures the pose rolls back and the diagonal replaces that
            # proposal from the same retained gradient and metric.
            #
            # WHY IT IS WORTH PORTING RATHER THAN TUNING. On MonoGS it learned
            # 6 on TUM fr1 and 2 on Replica room0 - 8.8% and 2.6% of their
            # frames - and forcing a fixed 20 on Replica cost two thirds of
            # the accuracy, 21% more iterations and 0.86 dB. On TUM a constant
            # would have done. So it earns its keep exactly where portability
            # is the problem, which is where Gaussian-SLAM now is.
            "PRE_ADAPTIVE_DIAG": ("adaptive_diag",
                                  lambda v: v not in ("0", "", "false", "False")),
            "PRE_DIAG_PATIENCE": ("adaptive_diag_patience", int),
            "PRE_DIAG_CALIB": ("adaptive_diag_calibration_frames", int),
            "PRE_DIAG_MIN": ("adaptive_diag_min_iter", int),
            "PRE_PROFILE_EVERY": ("profile_every", int),
            # Raw per-frame IT/FRAME/REL-GRAD/K series in summary(). Off by
            # default - lr_ladder.sh sets it for profiling/lr_proxy_replay.py.
            "LR_SERIES": ("log_series",
                          lambda v: v not in ("0", "", "false", "False")),
            "DRIFT_MULT": ("drift_mult", float),
            "DRIFT_WARMUP": ("drift_warmup", int),
            # THE ONE MOST LIKELY TO MATTER HERE. The record's ho0 failure ran
            # "EMPTY RENDERS 24202 iters over 398 frames - 61% of tracking
            # iterations rendering nothing", and every one of those decayed M
            # by beta2 with nothing added. 0.95^24202 is zero to any precision,
            # so P = M^{-1/2} was unbounded and no recovery was possible. On by
            # default; PRE_DEAD_FREEZE=0 restores the old path for an A/B.
            "PRE_DEAD_FREEZE": ("dead_freeze",
                                lambda v: v not in ("0", "", "false", "False")),
        }
        for _k, (_n, _c) in _envmap.items():
            if _k in os.environ:
                _pre[_n] = _c(os.environ[_k])
        _ps = {k: _pre.pop(k) for k in
               ("stop_rel_grad", "stop_min_iters", "stop_check_every",
                "stop_mode", "stop_patience", "handoff") if k in _pre}
        self.pre_stop_rel = float(_ps.get("stop_rel_grad", 0.0))
        self.pre_stop_min = int(_ps.get("stop_min_iters", 10))
        self.pre_stop_every = max(1, int(_ps.get("stop_check_every", 5)))
        self.pre_stop_mode = str(_ps.get("stop_mode", "rel"))
        self.pre_stop_patience = int(_ps.get("stop_patience", 10))
        self.pre_handoff = int(_ps.get("handoff", 0))
        self.pose_pre = None
        # The preceding frame's optimised relative world-to-camera transform.
        # Gaussian-SLAM parameterises W_t = W_{t-1} L_t and applies left
        # perturbations to L_t, so the local tangent basis changes whenever the
        # reference pose advances.  Kept on the host between track() calls and
        # moved to the metric's device only for the once-per-frame adjoint.
        self._previous_rel_w2c = None
        # Turns the committed pose's absolute error against ground truth into
        # a per-frame DELTA before it reaches note_frame_outcome() below - see
        # utils/online_lr_tuner.py's POSE_ERR section for why the delta, not
        # the level, is what the tuner's cap-bound fallback needs. Built
        # regardless of whether the preconditioner (or its tuner) is enabled,
        # since it is stateless-cheap and this keeps its lifetime tied to the
        # tracker rather than to a config combination that could change.
        self._pose_err_delta = PoseErrDelta()
        if _pre.pop("enabled", False):
            _pre.pop("dim", None)
            _lr_r = float(self.config["cam_rot_lr"])
            _lr_t = float(self.config["cam_trans_lr"])
            _diag_lr_t = _pre.pop("diag_trans_lr", None)
            _diag_lr_r = _pre.pop("diag_rot_lr", None)
            _auto = _pre.pop("auto_lr", False)
            _tgt = float(_pre.pop("target_step_norm", 0.0))
            if _auto and _tgt <= 0.0:
                # AUTO_LR IS NOT A SAFE DEFAULT - see the Replica section of
                # results/preconditioner_ladder.txt. It pins every step at
                # `target`, so the step can never shrink, and where the
                # achievable accuracy approaches that norm the optimiser orbits
                # at a radius equal to the answer. Correct only when the
                # accuracy floor sits well above the step size.
                _tgt = (3.0 * _lr_t ** 2 + 3.0 * _lr_r ** 2) ** 0.5
                print(f"[Preconditioner] auto step norm {_tgt:.3e} from "
                      f"cam_trans_lr={_lr_t:g}, cam_rot_lr={_lr_r:g}", flush=True)
            _has_diag_tail = (
                int(_pre.get("diag_after", 0)) > 0
                or bool(_pre.get("adaptive_diag", False))
            )
            if (_diag_lr_t is not None or _diag_lr_r is not None) and not _has_diag_tail:
                raise ValueError(
                    "PRE_DIAG_TRANS_LR/PRE_DIAG_ROT_LR require a diagonal "
                    "tail (PRE_DIAG_AFTER > 0 or PRE_ADAPTIVE_DIAG=1)"
                )
            if _diag_lr_t is not None or _diag_lr_r is not None:
                _diag_lr_t = _lr_t if _diag_lr_t is None else _diag_lr_t
                _diag_lr_r = _lr_r if _diag_lr_r is None else _diag_lr_r
                _pre["diag_lr"] = [_diag_lr_t] * 3 + [_diag_lr_r] * 3
                print(f"[Preconditioner] diagonal-tail rate override "
                      f"trans={_diag_lr_t:g}, rot={_diag_lr_r:g}", flush=True)
            elif _has_diag_tail and "diag_lr" not in _pre:
                # tangent_of_pose_delta and PosePreconditioner use [rho|theta],
                # so GSLAM's Adam-like diagonal refinement is translation
                # first and rotation last.  The full-matrix acquisition keeps
                # PRE_LR; only the diagonal tail receives these native rates.
                _pre["diag_lr"] = [_lr_t] * 3 + [_lr_r] * 3
                print(f"[Preconditioner] diagonal-tail rates "
                      f"trans={_lr_t:g}, rot={_lr_r:g}", flush=True)
            # ONLINE_LR_TUNE - the within-run alternative to the offline
            # two-run ladder (profiling/legacy/lr_ladder.sh). See
            # utils/online_lr_tuner.py. NO FIXED online_lr_budget is passed:
            # GSLAM's own num_iters can double mid-run on a high-loss frame
            # (see "Higher initial loss" above), so a single fixed budget
            # would misjudge which frames are truly at THEIR OWN cap. Instead
            # every frame reports at_cap and pose_err explicitly through
            # note_frame_outcome() below, using the same _iters_run >= num_iters
            # test and cam_trans_err this class already computes for its own
            # logging - this is a benchmarking harness, so the true trajectory
            # is already loaded for tracking, unlike a real no-ground-truth
            # deployment.
            _online_tune = bool(int(os.environ.get("ONLINE_LR_TUNE", "0")))
            if _online_tune:
                print("[Preconditioner] ONLINE_LR_TUNE=1 - the acquisition lr "
                     "will be discovered from this run's own it/frame, "
                     "falling back to pose error against ground truth on a "
                     "rung that is mostly at-cap, then held fixed.",
                     flush=True)
            self.pose_pre = PosePreconditioner(
                dim=6, lr=float(_pre.pop("lr", _lr_t)),
                target_step_norm=_tgt, device="cuda",
                online_lr_tune=_online_tune,
                online_lr_block_frames=int(
                    os.environ.get("ONLINE_LR_BLOCK", "12")),
                online_lr_max_halvings=int(
                    os.environ.get("ONLINE_LR_MAX_HALVINGS", "4")),
                online_lr_log_fn=lambda msg: print(msg, flush=True),
                **_pre)
            # TWO COMBINATIONS THE ADAPTIVE TRANSITION CANNOT HAVE, refused
            # here rather than producing a quietly wrong run.
            #
            # THE GRAPH. The switch decides from the frame's own loss BEFORE
            # applying each step, so it reads a scalar to the host every
            # iteration of the acquisition phase. That is a synchronisation,
            # which is illegal inside a captured region - and the loss would
            # in any case be read from a replayed buffer rather than the
            # current iteration. The sync disappears the moment a frame
            # switches to diagonal, which is why MonoGS tolerates it.
            #
            # THE HANDOFF. adaptive_diag already owns the phase transition;
            # a handoff would impose a second one on top of it.
            if self.pose_pre.adaptive_diag:
                if self.iter_graph.enabled:
                    raise ValueError(
                        "PRE_ADAPTIVE_DIAG needs a host read of the loss every "
                        "acquisition iteration, which cannot happen inside a "
                        "CUDA graph capture. Run with ITERGRAPH=0. The graph "
                        "changes execution and not arithmetic, so an ATE "
                        "measured without it is still comparable.")
                if self.pre_handoff > 0:
                    raise ValueError(
                        "PRE_ADAPTIVE_DIAG owns the phase transition; "
                        "PRE_HANDOFF would impose a second one. Use one.")
                print(f"[Preconditioner] ADAPTIVE diagonal transition: "
                      f"patience={self.pose_pre.adaptive_diag_patience}, "
                      f"calibration_frames="
                      f"{self.pose_pre.adaptive_diag_calibration_frames}, "
                      f"min_iter={self.pose_pre.adaptive_diag_min_iter}. "
                      f"The switch iteration is LEARNED per frame, not set.",
                      flush=True)
            if self.iter_graph.enabled:
                # GRAPH + PRECONDITIONER IS SUPPORTED, including with the
                # handoff, via phase-keyed capture - see
                # TrackingIterationGraph.run. The handoff makes a frame contain
                # two KINDS of iteration, so each phase warms up and captures
                # separately instead of one graph replaying the wrong kernels.
                #
                # step() is capture-safe by construction: branch-free,
                # in-place state, a device-side bias counter, and
                # se3_exp_capturable / mat_to_quat_capturable in place of the
                # host-branching originals. refactor() (the eigh) and the
                # plateau check's readback both run OUTSIDE the captured region.
                #
                # THE ECONOMICS ARE NOT THE SAME AS SINGLE-PHASE, THOUGH.
                # cudaGraphInstantiate is ~57 ms per capture and pays only over
                # enough replays - Gaussian-SLAM at ~195 replays/frame gave
                # 2.20x, MonoGS at ~74 gave -18%. A handoff pays it twice per
                # frame, and the preconditioner is simultaneously CUTTING the
                # replay count. PRE_HANDOFF=0 keeps it to one capture.
                print(f"[Preconditioner] iteration_graph is ON. "
                      f"{'Two captures per frame (handoff): ' if self.pre_handoff > 0 else 'One capture per frame. '}"
                      f"the ~57ms instantiate is amortised over fewer replays "
                      f"than the graph-only measurement assumed - compare "
                      f"against a graph-on baseline, not the graph-off one.",
                      flush=True)
            print(f"[Preconditioner] {self.pose_pre.summary()}", flush=True)
            if self.pre_stop_rel > 0.0 and self.pre_stop_mode == "best":
                print(f"[Preconditioner] PLATEAU stopping: no new best |g| for "
                      f"{self.pre_stop_patience} iters (margin "
                      f"{self.pose_pre.stop_improve:g}), floor "
                      f"{self.pre_stop_min}", flush=True)
            if self.pre_handoff > 0:
                print(f"[Handoff] preconditioner for iters "
                      f"0-{self.pre_handoff - 1}, then Adam", flush=True)

        # MODEL-INDEPENDENT WINDOWED CONVERGENCE. GSLAM and SplaTAM both
        # optimise an unnormalised quaternion plus translation, but the
        # quaternion radius is a loss-null gauge direction. The signal row
        # therefore carries the exact pre/post pose pair and _consume converts
        # it to the same six-dimensional [rho|theta] tangent used by the
        # preconditioner before this shared criterion sees it.
        _wc = dict(config.get("windowed_convergence", {}))
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
            "WCONV_Z": ("z_threshold", float),
            "WCONV_DECAY": ("decay_ratio", float),
            "WCONV_ENERGY_WINDOW": ("energy_window", int),
            "WCONV_ENERGY_PATIENCE": ("energy_patience", int),
            "WCONV_ENERGY_PHASE_RELATIVE": (
                "energy_phase_relative",
                lambda v: v not in ("0", "", "false", "False"),
            ),
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
            "WCONV_AUTO_FINAL_ONLY": ("auto_final_choice_only", lambda v: v not in ("0", "", "false", "False")),
        }
        for _key, (_name, _cast) in _wc_env.items():
            if _key in os.environ:
                _wc[_name] = _cast(os.environ[_key])
        _lr_t = float(self.config["cam_trans_lr"])
        _lr_r = float(self.config["cam_rot_lr"])
        if self.pose_pre is not None and self.pose_pre.diag_after > 0:
            # The diagonal phase is the only one long enough to propose in the
            # diag20 experiment (two complete 16-sample windows are required).
            # Use its actual per-coordinate rates, not PRE_LR repeated six
            # times, so translation and rotation have their native units.
            _wc_scales = self.pose_pre.diag_lr.detach().cpu().numpy()
        elif self.pose_pre is not None and self.pre_handoff <= 0:
            _wc_coord_scale = (
                float(self.pose_pre.target_step_norm) / np.sqrt(6.0)
                if float(self.pose_pre.target_step_norm) > 0.0
                else float(self.pose_pre.lr)
            )
            _wc_scales = [_wc_coord_scale] * 6
        else:
            # Adam or an Adam tail. tangent order is [translation|rotation].
            _wc_scales = [_lr_t] * 3 + [_lr_r] * 3
        _wc["budget"] = int(
            self.fixed_iters if self.fixed_iters > 0
            else self.config.get("iterations", 0)
        )
        self.windowed_stop = make_windowed_convergence(_wc, scales=_wc_scales)
        self.windowed_sweep = WindowedConvergenceSweep(
            os.environ.get("WCONV_SWEEP", ""), _wc, scales=_wc_scales
        )
        if (self.windowed_stop.active
                and (self.stopper.enabled or self.pre_stop_rel > 0.0)):
            raise ValueError(
                "active windowed convergence must own Gaussian-SLAM "
                "stopping: set ES=0 and STOP_REL=0"
            )

        # CLOSED-LOOP SUPERVISION FOR ACTIVE STOPPING. The local convergence
        # rule cannot see that an early commit changes mapping and keyframe
        # selection. This guard leaves the rule untouched but forces initial
        # calibration tails, audits every Nth eligible proposal, and vetoes
        # scale-free render-coverage / motion-prior outliers. It is GSLAM-side
        # control only; mapper and keyframe code receive the normal committed
        # pose and are not modified.
        _guard = dict(config.get("closed_loop_stop_guard", {}))
        _guard_env = {
            "WCONV_GUARD": ("enabled", lambda v: v not in ("0", "", "false", "False")),
            "WCONV_GUARD_CALIB": ("calibration_frames", int),
            "WCONV_GUARD_AUDIT": ("audit_every_proposals", int),
            "WCONV_GUARD_HISTORY": ("history_size", int),
            "WCONV_GUARD_MIN_HISTORY": ("min_history", int),
            "WCONV_GUARD_Z": ("robust_z", float),
            "WCONV_GUARD_COVERAGE": ("coverage_ratio", float),
            "WCONV_GUARD_CRITICAL": ("critical_coverage_ratio", float),
            "WCONV_GUARD_MOTION_P90": ("audit_motion_p90", float),
            "WCONV_GUARD_MOTION_MAX": ("audit_motion_max", float),
            "WCONV_GUARD_LOSS_P90": ("audit_loss_p90", float),
            "WCONV_GUARD_LOSS_MAX": ("audit_loss_max", float),
            # OFF BY DEFAULT (0). See closed_loop_stop_guard.py: N
            # consecutive clean latched audits release the full-budget
            # latch instead of it staying a one-way switch for the rest of
            # the run.
            "WCONV_GUARD_RELEASE_AFTER": ("release_after", int),
        }
        for _key, (_name, _cast) in _guard_env.items():
            if _key in os.environ:
                _guard[_name] = _cast(os.environ[_key])
        self.closed_loop_guard = AuditedStopGuard(_guard)
        # GRADIENT REUSE. Same flags as SplaTAM - GRAD_REUSE,
        # GRAD_REUSE_WARMUP, GRAD_REUSE_NOACC - so one sweep drives both
        # models and an arm cannot be on for one and off for the other
        # without the run name saying so.
        #
        # WARMUP 6 IS A SplaTAM/TUM RESULT, CARRIED HERE AS A HYPOTHESIS.
        # It works there because the pose moves roughly twice as fast in
        # the opening iterations of a frame as it does later, and
        # protecting those is where the whole ATE recovery came from. This
        # model tracks on a different budget with a different schedule, so
        # neither the number nor the mechanism transfers by default.
        # GRADSTALE=1 measures it here rather than assuming it.
        _gr_cfg = grad_reuse_env(
            config.get("tracking", {}).get("grad_reuse", {})
        )
        _gr_cal_frames = int(os.environ.get(
            "GRAD_REUSE_CALIBRATE_FRAMES", "0") or 0
        )
        if _gr_cal_frames:
            # The coupled full-matrix acquisition must be built entirely from
            # fresh gradients. Calibrated mode therefore owns both edges of
            # the reuse window: warmup is the fixed DIAG transition, while
            # cooldown is learned from the no-reuse run-in frames.
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
            if "GRAD_REUSE_WARMUP" in os.environ:
                # EXPLICIT OVERRIDE OF THE DEFAULT DIAG-TIED WARMUP. The
                # default (warmup == diag_after) exists so the full-matrix
                # acquisition never sees a reused gradient before the
                # diagonal-to-full transition. A fixed, smaller warmup is a
                # deliberate hypothesis test against that default - that
                # per-frame gradient correlation is already high a few
                # iterations in, well before diag_after, so protecting the
                # whole diagonal tail is unnecessary. Not validated; compare
                # ATE against the diag-tied default before trusting it.
                _warmup = int(os.environ["GRAD_REUSE_WARMUP"])
                print("[GradReuse] calibrated mode: explicit "
                      f"GRAD_REUSE_WARMUP={_warmup} overrides the default "
                      f"DIAG={_diag_after} transition", flush=True)
            else:
                _warmup = _diag_after
            if "GRAD_REUSE_COOLDOWN" in os.environ:
                print("[GradReuse] calibrated mode ignores the fixed "
                      f"GRAD_REUSE_COOLDOWN={os.environ['GRAD_REUSE_COOLDOWN']}; "
                      "the first-frame stopping distribution selects it",
                      flush=True)
            _gr_cfg["warmup"] = _warmup
            _gr_cfg["cooldown"] = 0
            _gr_cfg["calibration_frames"] = _gr_cal_frames
            _gr_cfg["calibration_margin"] = int(os.environ.get(
                "GRAD_REUSE_CALIBRATE_MARGIN", "15") or 15
            )
            _gr_cfg["calibration_round"] = int(os.environ.get(
                "GRAD_REUSE_CALIBRATE_ROUND", "10") or 10
            )
            _gr_cfg["calibration_stat"] = os.environ.get(
                "GRAD_REUSE_CALIBRATE_STAT", "min"
            )
            # GSLAM's closed-loop guard deliberately sends its initial
            # calibration/audit frames to the full budget. Those are not a
            # stopping location and must not pull the reuse ceiling upward.
            _gr_cfg["calibration_exclude_reasons"] = ("cap",)
            print("[GradReuse] automatic window calibration: reuse OFF for "
                  f"the first {_gr_cal_frames} tracked frames; warmup="
                  f"{_warmup}; cooldown=floor(("
                  f"{_gr_cfg['calibration_stat']} actual stop - "
                  f"{_gr_cfg['calibration_margin']}) / "
                  f"{_gr_cfg['calibration_round']}) * "
                  f"{_gr_cfg['calibration_round']}; cap frames are excluded "
                  "from the statistic", flush=True)
        self.grad_reuse = GradReuse(_gr_cfg)
        if (self.grad_reuse.calibration_frames
                and not self.windowed_stop.active):
            raise ValueError(
                "GRAD_REUSE_CALIBRATE_FRAMES requires an active WCONV "
                "stopping criterion during its run-in frames"
            )
        if (self.grad_reuse.enabled and self.iter_graph.enabled
                and self.iter_graph.warmup_iters > self.grad_reuse.warmup):
            # THE TWO NOW COMPOSE, via iter_graph.run(bypass=...): reuse
            # iterations run eager and rendering ones replay. What still
            # has to hold is that the CAPTURE lands on a rendering
            # iteration - bypass returns before _warmup_done is touched,
            # so the graph warms up on rendering iterations only, and it
            # needs enough of them to exist before the reuse band opens.
            raise ValueError(
                "iteration_graph.warmup_iters (%d) must be <= "
                "grad_reuse.warmup (%d), or the graph would still be "
                "warming up when reuse starts and the capture could be "
                "starved of rendering iterations"
                % (self.iter_graph.warmup_iters, self.grad_reuse.warmup))
        self._render_clock = (
            self.grad_reuse.enabled
            and os.environ.get("GRAD_REUSE_STOP_CLOCK", "") == "render"
        )
        if self._render_clock:
            print("[Tracking] windowed stopper runs on the RENDER clock "
                  "(reused iterations are skipped and their steps banked)",
                  flush=True)
        self._windowed_scales = np.asarray(_wc_scales, dtype=np.float64)
        if self.closed_loop_guard.enabled and not self.windowed_stop.active:
            raise ValueError(
                "closed_loop_stop_guard requires active windowed convergence"
            )

        # opt_cam_rot is a 4-element quaternion and opt_cam_trans a 3-vector.
        # The optional extra pair is appended after the unchanged candidate
        # pose columns, so existing early stopping and pose selection retain
        # their byte-for-byte row layout when WCONV is off.
        _es_cfg = dict(config.get("es_signals", {}))
        if self.windowed_stop.enabled:
            _es_cfg["enabled"] = True
            _es_cfg["batch"] = int(os.environ.get("WCONV_BATCH", "8"))
        self.es_signals = ESSignalBuffer(
            _es_cfg, include_pose_pair=self.windowed_stop.enabled,
            extra_cols=(1 if self.closed_loop_guard.enabled else 0),
        )
        if (self.windowed_stop.active
                and self.windowed_stop.check_every % self.es_signals.batch != 0):
            raise ValueError(
                "active windowed_convergence check_every must be a multiple "
                "of es_signals.batch so its decision lands on a drain boundary"
            )
        print(f"[Tracking] {self.windowed_stop.summary()}", flush=True)
        print(f"[Tracking] {self.closed_loop_guard.summary()}", flush=True)
        # GRADIENT STALENESS PROBE. Diagnostic only - it adds an extra
        # forward+backward at every probe point, so wall time, in-loop
        # time and ms/iter are all meaningless on a run with it on.
        #
        # WHY IT IS BEING WIRED HERE AT ALL. The cos 0.9473 / mag 0.942
        # that the whole gradient-reuse mechanism rests on was measured on
        # SplaTAM at iterations 5 and 15 of a ~27-iteration frame. GSLAM
        # frames run ~132 iterations, and nobody has measured staleness in
        # its polish band. That number decides the maximum viable cooldown:
        # reuse breaks even only when a reused step delivers q > c_u/c_r =
        # 0.49 of a fresh step, and the prize runs from -6.6% at the
        # current window to about -22% at the widest one.
        #
        # GRADSTALE=1 enables it. GRADSTALE_STARTS / _OFFSETS / _FRAMES
        # override the schedule, comma separated.
        _gs = dict(config.get("tracking", {}).get("grad_staleness", {}))
        if os.environ.get("GRADSTALE", "0") not in ("0", "", "false", "False"):
            _gs["enabled"] = True
        for _k, _e in (("starts", "GRADSTALE_STARTS"),
                       ("offsets", "GRADSTALE_OFFSETS"),
                       ("frames", "GRADSTALE_FRAMES")):
            _v = os.environ.get(_e)
            if _v:
                _gs[_k] = [int(x) for x in _v.split(",") if x.strip()]
        self.stale_probe = GradStalenessProbe(_gs)
        # PER-ITERATION TRACE for profiling/plot_iter_trace.py (ITER_TRACE=
        # or tracking.iter_trace). Which frames are traced is decided on the
        # host per frame, so a captured iteration would freeze the first
        # frame's decision - refused rather than traced wrong.
        self.iter_trace = IterTrace(config.get("iter_trace", {}))
        if self.iter_trace.enabled and self.iter_graph.enabled:
            raise ValueError(
                "ITER_TRACE cannot run with the iteration graph: the "
                "per-frame trace switch is host Python. Set ITERGRAPH=0.")
        if self.stale_probe.enabled and self.iter_graph.enabled:
            # The probe runs an isolated backward from the HOST, between
            # iterations. Under capture that lands inside the recorded
            # region on the capture iteration and is replayed forever
            # after; and it holds tensors across the boundary, which is
            # the live-reference hazard that already cost one debugging
            # session here. It is a diagnostic run, so refuse rather than
            # measure something subtly wrong.
            raise ValueError(
                "GRADSTALE cannot run with the iteration graph: the "
                "probe needs an isolated host-side backward. "
                "Set ITERGRAPH=0.")
        if self.stale_probe.enabled and self.grad_reuse.enabled:
            # The probe measures cos(g0, g_true) at each offset, and
            # g_true has to be a REAL gradient at that iteration. A reuse
            # iteration computes none, so half the offsets would compare
            # g0 against a copy of itself and read a flattering 1.0.
            raise ValueError(
                "GRADSTALE cannot run with grad_reuse: the probe needs a "
                "fresh gradient at every offset. Set GRAD_REUSE=0.")
        print(f"[Tracking] {self.grad_reuse.summary()}", flush=True)
        for _line in self.windowed_sweep.summary_lines():
            print(f"[Tracking] {_line}", flush=True)
        # Tracking-only wall time. SLAM_WALL_TIME covers the whole run, mapping
        # included, and the iteration graph cannot touch mapping - so a
        # graph-vs-eager comparison on SLAM_WALL_TIME understates the tracking
        # effect by however much mapping contributes, and cannot attribute it
        # either. MonoGS has had _tracking_time_total for a while; this is the
        # equivalent, added after the first Gaussian-SLAM A/B came back
        # 253.3s vs 355.7s with no way to say how much of that was tracking.
        self._tracking_time_total = 0.0
        self._tracking_iters_total = 0
        self._tracking_frames = 0
        self._iters_per_frame = []
        self._budget_per_frame = []
        # PER-FRAME TRACKING TIME, logging only. The running total cannot say
        # whether a stopping rule's cost is per-iteration or per-frame: the
        # GSLAM/TUM control cut 33% of iterations and moved tracking time by
        # -1.3%, and a two-point fit across separate runs puts the fixed term
        # anywhere from ~3 to ~600 ms under the stated timing noise. Regressing
        # time on iterations across one run's ~590 frames separates the two.
        # Appended beside _iters_per_frame so the series cannot misalign.
        self._track_ms_per_frame = []
        if self.iter_graph.enabled:
            self._preflight_iteration_graph()

    def _iteration_distribution(self) -> str:
        """How the frame budget was actually spent, not just its mean.

        A mean of 178/200 is consistent with every frame stopping at 178 and
        with a quarter of frames stopping at 118 while the rest run the cap.
        Only the second says the criterion is selective, and only the second
        justifies the machinery. Reported as counts and percentiles over the
        frames that DID stop, since those are the ones the rule acted on.
        """
        n = len(self._iters_per_frame)
        if n == 0:
            return ""
        full = [i for i, b in zip(self._iters_per_frame, self._budget_per_frame)
                if i >= b]
        stopped = [i for i, b in zip(self._iters_per_frame, self._budget_per_frame)
                   if i < b]
        budgets = sorted(set(self._budget_per_frame))
        _b = (f"{budgets[0]}" if len(budgets) == 1
              else f"{min(budgets)}-{max(budgets)} (adaptive)")
        out = (f"Iterations per frame: budget {_b}, "
               f"{len(full)}/{n} frames ran the FULL budget "
               f"({100.0 * len(full) / n:.1f}%), {len(stopped)} stopped early")
        if stopped:
            q = sorted(stopped)
            def _p(f):
                return q[min(len(q) - 1, int(f * (len(q) - 1) + 0.5))]
            saved = sum(b - i for i, b in
                        zip(self._iters_per_frame, self._budget_per_frame) if i < b)
            out += (f"; stopped frames p10/med/p90 = "
                    f"{_p(0.10)}/{_p(0.50)}/{_p(0.90)}, min {q[0]}, "
                    f"total saved {saved} iterations "
                    f"({saved / n:.1f}/frame averaged over ALL frames, "
                    f"{saved / len(stopped):.1f} over the ones that stopped)")
        return out

    def tracking_summary(self) -> str:
        if self._tracking_iters_total == 0:
            return "Tracking time: no frames tracked"
        lines = [
            (f"Tracking time: {self._tracking_time_total:.1f}s over "
             f"{self._tracking_iters_total} iterations across "
             f"{self._tracking_frames} frames "
             f"({1000 * self._tracking_time_total / self._tracking_iters_total:.2f} "
             f"ms/iter, "
             f"{self._tracking_iters_total / self._tracking_frames:.1f} iters/frame)"),
        ]
        # Convergence/guard calibration deliberately runs full tracking tails,
        # while gradient-reuse calibration deliberately keeps reuse disabled.
        # Including that shared prefix makes the installed configuration look
        # slower than its steady state.  Preserve the all-frame aggregate above
        # for wall-time accounting, then report a like-for-like steady-state
        # aggregate for experiment tables.  The per-frame iteration and time
        # arrays are appended together, so slicing both at the same boundary
        # cannot mix calibration work into the post-calibration rate.
        _convergence_calibration = 0
        if (getattr(self.windowed_stop, "enabled", False)
                and hasattr(self.windowed_stop, "calibration_frames")):
            _convergence_calibration = (
                int(getattr(
                    self.windowed_stop, "calibration_skip_frames", 0
                ))
                + int(self.windowed_stop.calibration_frames)
            )
        _reuse_calibration = (
            int(getattr(self.grad_reuse, "calibration_frames", 0))
            if getattr(self.grad_reuse, "enabled", False) else 0
        )
        _guard_calibration = (
            int(getattr(self.closed_loop_guard, "calibration_frames", 0))
            if getattr(self.closed_loop_guard, "enabled", False) else 0
        )
        # These run-ins overlap at the start of the sequence.  Excluding their
        # maximum, not their sum, begins the reported interval only after every
        # installed policy is in steady state without discarding valid frames.
        _calibration_prefix = min(
            max(
                _convergence_calibration,
                _reuse_calibration,
                _guard_calibration,
                0,
            ),
            len(self._iters_per_frame),
        )
        if _calibration_prefix:
            _steady_iters = self._iters_per_frame[_calibration_prefix:]
            _steady_ms = self._track_ms_per_frame[_calibration_prefix:]
            _calibration_detail = (
                f"convergence={_convergence_calibration}, "
                f"reuse={_reuse_calibration}, guard={_guard_calibration}"
            )
            if _steady_iters:
                _steady_total_iters = sum(_steady_iters)
                _steady_total_ms = sum(_steady_ms)
                lines.append(
                    f"Tracking post-calibration time: "
                    f"{_steady_total_ms / 1000.0:.1f}s over "
                    f"{_steady_total_iters} iterations across "
                    f"{len(_steady_iters)} frames "
                    f"({(_steady_total_ms / _steady_total_iters):.2f} "
                    f"ms/iter, "
                    f"{(_steady_total_iters / len(_steady_iters)):.1f} "
                    f"iters/frame; excluded first {_calibration_prefix} "
                    f"calibration frames; {_calibration_detail})"
                )
            else:
                lines.append(
                    "Tracking post-calibration time: unavailable "
                    f"(excluded first {_calibration_prefix} calibration "
                    f"frames; {_calibration_detail}; no later frames)"
                )
        # DIRECTLY UNDER the mean it qualifies, so the two cannot be read apart.
        _dist = self._iteration_distribution()
        if _dist:
            lines.append(_dist)
        # Both series from the same per-frame append, so index i is one frame
        # in both. Read by profiling/frame_cost_fit.py.
        if self._track_ms_per_frame:
            lines.append(
                f"FRAME ITERS n={len(self._iters_per_frame)}: "
                + ",".join(str(i) for i in self._iters_per_frame))
            lines.append(
                f"FRAME TRACK MS n={len(self._track_ms_per_frame)}: "
                + ",".join(f"{v:.1f}" for v in self._track_ms_per_frame))
        if os.environ.get("GSLAM_SKIP_VIS", "0") != "1":
            lines.append(self.pose_pre.summary() if self.pose_pre is not None
                         else "Pose preconditioner: disabled")
            if (self.pose_pre is not None
                    and os.environ.get(
                        "GSLAM_PRINT_STEP_PROFILE", "1"
                    ) not in ("0", "", "false", "False")
                    and self.pose_pre.step_profile()):
                lines.append(self.pose_pre.step_profile())
        # SPARSE SAMPLING, next to the iteration counts it has to be read
        # against. A masked-iteration count with no forced-dense count beside
        # it cannot distinguish "engaged and committed cleanly" from "engaged
        # and committed from a masked render", and the second looks identical
        # in every other column.
        if self._ps_frames:
            lines.append(
                f"Tile-mask pixel sampling: {self._ps_iters_masked} iters masked "
                f"over {self._ps_frames} frames at ratio "
                f"{self.pixel_sample_cfg['sample_ratio']} (always-sparse); "
                f"{self._ps_dense_commits}/{self._ps_frames} frames got a forced "
                f"dense iteration before commit; masked losses and coverage "
                f"rescaled onto the dense scale"
                + ("" if self._ps_dense_commits >= self._ps_frames else
                   " <- SHORTFALL: both the stop path and the cap arm the "
                   "forced-dense iteration, so a frame reaching neither is a "
                   "bug, and its committed pose came from a MASKED render"))
        elif os.environ.get("VERBOSE_DIAG", "0") not in ("0", "", "false", "False"):
            lines.append("Tile-mask pixel sampling: disabled")
        budget = (self.fixed_iters if self.fixed_iters > 0
                  else self.config.get("iterations", 0))
        if os.environ.get("GSLAM_SKIP_VIS", "0") != "1":
            lines.append(self.windowed_stop.summary(budget))
            lines.append(self.closed_loop_guard.summary())
        lines.append(self.grad_reuse.summary())
        if self.stale_probe.enabled:
            lines.append(self.stale_probe.summary())
        if os.environ.get("GSLAM_SKIP_VIS", "0") != "1":
            lines.extend(
                self.windowed_sweep.summary_lines(budget)
            )
        return "\n".join(lines)

    def _preflight_iteration_graph(self):
        """Fail at startup, not at frame 300.

        Gaussian-SLAM has no early stopping and no tile masking wired in, so
        this is a shorter list than MonoGS's - but the binning prerequisite is
        the same, and it is the one that produces a hard capture failure rather
        than a silently wrong run.
        """
        if not self.binning_capacity.enabled:
            raise ValueError(
                "iteration_graph.enabled requires binning_capacity.enabled. "
                "Without a fixed capacity the rasterizer reads num_rendered "
                "back to the host to size its binning buffer, which is illegal "
                "during capture - the first capture attempt will fail."
            )
        if self.binning_capacity.warmup_iters > self.iter_graph.warmup_iters:
            raise ValueError(
                f"binning_capacity.warmup_iters "
                f"({self.binning_capacity.warmup_iters}) must be <= "
                f"iteration_graph.warmup_iters ({self.iter_graph.warmup_iters}). "
                f"The capacity has to be frozen BEFORE the capture iteration "
                f"runs, or the capture records the host readback path it is "
                f"supposed to be replacing."
            )
        if self.stopper.enabled:
            _budget = int(self.config.get("iterations", 0))
            if _budget and self.stopper.min_iters >= _budget:
                raise ValueError(
                    f"early_stop.min_iters={self.stopper.min_iters} is >= "
                    f"tracking iterations={_budget}, so early stopping can "
                    f"never fire while still logging enabled=True. Rescale it "
                    f"for this model's budget, and move "
                    f"retune_min_iters_floor with it or the first retune's "
                    f"elbow detection drags min_iters to the class default of 8."
                )
            if self.stopper.min_iters <= self.iter_graph.warmup_iters:
                print(
                    f"[Tracker] WARNING: early stopping can fire at iteration "
                    f"{self.stopper.min_iters}, at or before the graph's warmup "
                    f"of {self.iter_graph.warmup_iters}. Frames stopping that "
                    f"early never reach a capture, so frames_captured "
                    f"under-reports.",
                    flush=True,
                )

    def compute_losses(self, gaussian_model: GaussianModel, render_settings: dict,
                       opt_cam_rot: torch.Tensor, opt_cam_trans: torch.Tensor,
                       gt_color: torch.Tensor, gt_depth: torch.Tensor, depth_mask: torch.Tensor,
                       binning_kwargs: dict = None,
                       tile_mask: torch.Tensor = None,
                       loss_scale: torch.Tensor = None) -> tuple:
        """ Computes the tracking losses with respect to ground truth color and depth.
        Args:
            gaussian_model: The current state of the Gaussian model of the scene.
            render_settings: Dictionary containing rendering settings such as image dimensions and camera intrinsics.
            opt_cam_rot: Optimizable tensor representing the camera's rotation.
            opt_cam_trans: Optimizable tensor representing the camera's translation.
            gt_color: Ground truth color image tensor.
            gt_depth: Ground truth depth image tensor.
            depth_mask: Binary mask indicating valid depth values in the ground truth depth image.
        Returns:
            A tuple containing losses and renders
        """
        # `torch.eye(4).cuda()` and `torch.ones(N, 1).cuda()` build on the CPU
        # and then copy host-to-device. A pageable H2D copy is synchronous, so
        # both are capture-fatal - and the second one copied a full
        # (num_gaussians, 1) tensor across the bus on EVERY tracking iteration,
        # which was never useful work even without a graph.
        #
        # Constructing directly on the device is a device-side fill with the
        # scalar as a kernel argument: no host memory involved. Note the
        # contrast with `x[i] = 1.0`, which IS a host round trip - see
        # MonoGS/gaussian_splatting/utils/graphics_utils.py for that one.
        rel_transform = torch.eye(4, device="cuda", dtype=torch.float32)
        rel_transform[:3, :3] = build_rotation(F.normalize(opt_cam_rot[None]))[0]
        rel_transform[:3, 3] = opt_cam_trans

        pts = gaussian_model.get_xyz()
        pts_ones = torch.ones(pts.shape[0], 1, device="cuda", dtype=torch.float32)
        pts4 = torch.cat((pts, pts_ones), dim=1)
        transformed_pts = (rel_transform @ pts4.T).T[:, :3]

        quat = F.normalize(opt_cam_rot[None])
        _rotations = multiply_quaternions(gaussian_model.get_rotation(), quat.unsqueeze(0)).squeeze(0)

        render_dict = render_gaussian_model(gaussian_model, render_settings,
                                            override_means_3d=transformed_pts, override_rotations=_rotations,
                                            tracking_only=self.tracking_only_backward,
                                            binning_kwargs=binning_kwargs,
                                            tile_mask=tile_mask)
        rendered_color, rendered_depth = render_dict["color"], render_dict["depth"]
        alpha_mask = render_dict["alpha"] > self.alpha_thre

        tracking_mask = torch.ones_like(alpha_mask).bool()
        tracking_mask &= depth_mask
        depth_err = torch.abs(rendered_depth - gt_depth) * depth_mask

        if self.filter_alpha:
            tracking_mask &= alpha_mask
        if self.filter_outlier_depth:
            # `if ... and torch.median(depth_err) > 0` was BOTH a host sync and
            # data-dependent control flow - the worse of the two problems for a
            # CUDA graph. A sync merely fails the capture; frozen control flow
            # succeeds, and then replays whichever branch the capture iteration
            # happened to take for every remaining iteration of the frame.
            #
            # The `| (median <= 0)` term reproduces the original guard exactly:
            # when the median is 0 the old code skipped the filter entirely,
            # i.e. kept every pixel, which is what an all-True term does here.
            # Also computes the median ONCE instead of twice.
            _median = torch.median(depth_err)
            tracking_mask &= (depth_err < 50 * _median) | (_median <= 0)

        color_loss = l1_loss(rendered_color, gt_color, agg="none")
        depth_loss = l1_loss(rendered_depth, gt_depth, agg="none") * tracking_mask

        if self.soft_alpha:
            alpha = render_dict["alpha"] ** 3
            color_loss *= alpha
            depth_loss *= alpha
            if self.mask_invalid_depth_in_color_loss:
                color_loss *= tracking_mask
        else:
            color_loss *= tracking_mask

        color_loss = color_loss.sum()
        depth_loss = depth_loss.sum()

        # THE LOSSES ARE ABSOLUTE SUMS, NOT MEANS, so they scale with the
        # number of pixels rendered. Under a tile mask that is ~sample_ratio
        # of the image, which makes a masked loss and a dense loss
        # incomparable for reasons that have nothing to do with pose error:
        #
        #   * the committed pose is the BEST-LOSS candidate, so a dense
        #     candidate carrying a 1/ratio larger sum can essentially never
        #     win - the forced-dense iteration did the exact opposite of
        #     what it was added for;
        #   * incumbent-energy convergence reads the loss series, which took
        #     a step change the moment the frame flipped to dense;
        #   * the adaptive-diagonal arm scores proposals by loss improvement,
        #     and a proposal straddling the switch always reads as a failure.
        #
        # Scaling the MASKED loss up by 1/kept_fraction puts both on the
        # dense scale. The dense path passes loss_scale=None and is left byte
        # for byte as it was, so every existing GSLAM number is untouched.
        # SplaTAM and MonoGS normalise their tracking loss and so never
        # needed this - which is why the port did not carry it across.
        if loss_scale is not None:
            color_loss = color_loss * loss_scale
            depth_loss = depth_loss * loss_scale

        return color_loss, depth_loss, rendered_color, rendered_depth, alpha_mask

    def track(self, frame_id: int, gaussian_model: GaussianModel, prev_c2ws: np.ndarray) -> np.ndarray:
        """
        Updates the camera pose estimation for the current frame based on the provided image and depth, using either ground truth poses,
        constant speed assumption, or visual odometry.
        Args:
            frame_id: Index of the current frame being processed.
            gaussian_model: The current Gaussian model of the scene.
            prev_c2ws: Array containing the camera-to-world transformation matrices for the frames (0, i - 2, i - 1)
        Returns:
            The updated camera-to-world transformation matrix for the current frame.
        """
        _, image, depth, gt_c2w = self.dataset[frame_id]

        if (self.help_camera_initialization or self.odometry_type == "odometer") and self.odometer.last_rgbd is None:
            _, last_image, last_depth, _ = self.dataset[frame_id - 1]
            self.odometer.update_last_rgbd(last_image, last_depth)

        if self.odometry_type == "gt":
            return gt_c2w
        elif self.odometry_type == "const_speed":
            init_c2w = extrapolate_poses(prev_c2ws[1:])
        elif self.odometry_type == "odometer":
            odometer_rel = self.odometer.estimate_rel_pose(image, depth)
            init_c2w = prev_c2ws[-1] @ odometer_rel

        last_c2w = prev_c2ws[-1]
        last_w2c = np.linalg.inv(last_c2w)
        init_rel = init_c2w @ np.linalg.inv(last_c2w)
        init_rel_w2c = np.linalg.inv(init_rel)
        reference_w2c = last_w2c
        render_settings = get_render_settings(
            self.dataset.width, self.dataset.height, self.dataset.intrinsics, reference_w2c)
        opt_cam_rot, opt_cam_trans = compute_camera_opt_params(init_rel_w2c)
        gaussian_model.training_setup_camera(opt_cam_rot, opt_cam_trans, self.config,
                                             capturable=self.iter_graph.enabled)
        if self.pose_pre is not None:
            # CARRIES M, RESETS m and the plateau state. Gaussian-SLAM does
            # not optimise an absolute W_t pose: its variable is the relative
            # transform L_t in W_t = W_{t-1} L_t. A left perturbation xi in
            # the old local variable and the equivalent perturbation xi' in
            # the next one obey
            #
            #   Ad_{W_{t-1}} xi = Ad_{W_t} xi'
            #   xi' = Ad_{W_t^-1 W_{t-1}} xi = Ad_{L_t^-1} xi.
            #
            # Therefore the previous frame's best L_t supplies the exact
            # change of tangent basis. Reusing M numerically (transport=None)
            # mixes two relative coordinate systems and makes the first update
            # of each frame the most likely excursion.
            _A = None
            if self._previous_rel_w2c is not None:
                with torch.no_grad():
                    _prev_rel = self._previous_rel_w2c.to(
                        device=self.pose_pre.M.device,
                        dtype=self.pose_pre.M.dtype,
                    )
                    _A = se3_adjoint(torch.linalg.inv(_prev_rel))
            # WINDOWED PROFILE, before carry_frame clears the frame state. A
            # cumulative profile over 590 frames averages the window where the
            # tracker breaks into the hundreds where it does not.
            if self.pose_pre.profile_due():
                print(f"[Preconditioner] {self.pose_pre.summary()}", flush=True)
                _sp = self.pose_pre.step_profile()
                if _sp:
                    print(_sp, flush=True)
                self.pose_pre.reset_profile()
            self.pose_pre.carry_frame(transport=_A)

        gt_color = self.transform(image).cuda()
        gt_depth = np2torch(depth, "cuda")
        depth_mask = gt_depth > 0.0
        # ScanNet marks poses that its offline reconstruction could not solve
        # with non-finite values.  They are ground truth used only for the
        # diagnostic errors below; tracking itself is initialized from the
        # odometer/constant-speed estimate.  Keep the RGB-D frame in the SLAM
        # sequence and record an unavailable diagnostic as NaN, matching the
        # evaluator's existing contract (it filters non-finite GT poses from
        # ATE).  In particular, scene0059_00 contains such a block at frames
        # 1460--1479.
        gt_pose_valid = np.isfinite(gt_c2w).all()
        if gt_pose_valid:
            try:
                gt_quat_np = R.from_matrix(
                    gt_c2w[:3, :3]).as_quat(canonical=True)[[3, 0, 1, 2]]
            except (ValueError, np.linalg.LinAlgError):
                # A finite but singular/malformed rotation is equally
                # unusable as a ground-truth diagnostic.
                gt_pose_valid = False

        if gt_pose_valid:
            gt_trans = np2torch(gt_c2w[:3, 3])
            gt_quat = np2torch(gt_quat_np)
        else:
            print(f"[GT pose] frame {frame_id} is invalid; logging NaN so it "
                  "is filtered from ATE", flush=True)
            gt_trans = torch.full((3,), float("nan"))
            gt_quat = torch.full((4,), float("nan"))
        num_iters = self.fixed_iters if self.fixed_iters > 0 else self.config["iterations"]
        current_min_loss = float("inf")

        print(f"\nTracking frame {frame_id}")
        # Initial loss check
        color_loss, depth_loss, _, _, _ = self.compute_losses(gaussian_model, render_settings, opt_cam_rot,
                                                              opt_cam_trans, gt_color, gt_depth, depth_mask)
        if len(self.frame_color_loss) > 0 and (
            color_loss.item() > self.init_err_ratio * np.median(self.frame_color_loss)
            or depth_loss.item() > self.init_err_ratio * np.median(self.frame_depth_loss)
        ):
            # SKIPPED under TRACK_ITERS - see the constructor. The odometry
            # re-initialisation below is NOT skipped: it is a pose fix, not a
            # budget change, and dropping it would make the clamp a second
            # variable instead of one.
            if self.fixed_iters <= 0:
                num_iters *= 2
                print(f"Higher initial loss, increasing num_iters to {num_iters}")
            if self.help_camera_initialization and self.odometry_type != "odometer":
                _, last_image, last_depth, _ = self.dataset[frame_id - 1]
                self.odometer.update_last_rgbd(last_image, last_depth)
                odometer_rel = self.odometer.estimate_rel_pose(image, depth)
                init_c2w = last_c2w @ odometer_rel
                init_rel = init_c2w @ np.linalg.inv(last_c2w)
                init_rel_w2c = np.linalg.inv(init_rel)
                opt_cam_rot, opt_cam_trans = compute_camera_opt_params(init_rel_w2c)
                gaussian_model.training_setup_camera(opt_cam_rot, opt_cam_trans, self.config,
                                             capturable=self.iter_graph.enabled)
                render_settings = get_render_settings(
                    self.dataset.width, self.dataset.height, self.dataset.intrinsics, last_w2c)
                print(f"re-init with odometer for frame {frame_id}")

        # reset() also drives EarlyStop's retune bookkeeping - it counts
        # frames, opens probe windows and fires _do_retune() - so it must be
        # called once per frame regardless of whether stopping is enabled.
        self.stopper.reset()
        self.binning_capacity.reset_for_frame()
        self.iter_graph.reset_for_frame()
        self.es_signals.reset_for_frame()
        self.windowed_stop.reset_frame()
        self.windowed_sweep.reset_frame()
        self.closed_loop_guard.reset_frame()
        _clock = RenderClock() if self._render_clock else None

        # Batched candidate selection. The best-loss pose is tracked on the
        # HOST from the drained rows, in exactly the plain-Python comparison
        # the eager path uses - only the FREQUENCY of the host read changes,
        # never where the comparison happens. The on-device torch.where variant
        # was tried on SplaTAM (sync_free_candidates), broke tracking at ATE
        # 64.28cm, and its root cause was never found.
        _best_loss_host = float("inf")
        _best_rot_host = _best_tran_host = None
        _best_coverage_host = None
        _initial_coverage_host = None

        def _consume(rows):
            """Feed drained iterations to the candidate and the stopper.

            Returns True if early stopping fired. Rows after the firing one are
            dropped: they were executed (the stop is detected up to batch-1
            iterations late) but the eager path would never have run them, so
            ignoring them keeps the decisions identical and makes the batching
            a pure performance change.
            """
            nonlocal _best_loss_host, _best_rot_host, _best_tran_host
            nonlocal _best_coverage_host, _initial_coverage_host
            # ONE BATCHED TANGENT PER DRAIN, NOT ONE CALL PER ROW. The per-row call
            # dispatched a few dozen tiny float64 CPU ops for every counted iteration
            # while the GPU waited: ~1.15-1.33 ms/row on CPU, a large share of why
            # stopping cut 31% of GSLAM/fr1_desk's iterations but ~5% of its tracking
            # time. tangent_of_pose_delta_batch is pinned to the single-row result by
            # utils/test_tangent_batch.py (max difference 0.0 across 1e-5..1 rad), so
            # every stopping decision is unchanged. Rows after a stop are still
            # ignored; computing their tangents up front has no side effects.
            _steps = None
            if self.windowed_stop.enabled and rows:
                _steps = tangent_of_pose_delta_batch(
                    torch.stack([_r["pre_rot"] for _r in rows]),
                    torch.stack([_r["pre_tran"] for _r in rows]),
                    torch.stack([_r["post_rot"] for _r in rows]),
                    torch.stack([_r["post_tran"] for _r in rows]),
                ).detach().cpu().numpy()
            for _j, _r in enumerate(rows):
                _coverage = (
                    float(_r["extra"][0])
                    if self.closed_loop_guard.enabled else None
                )
                if _initial_coverage_host is None and _coverage is not None:
                    _initial_coverage_host = _coverage
                _window_proposed = False
                if self.windowed_stop.enabled:
                    _step = _steps[_j]
                    _window_it = _r["iter"]
                    _window_skip = False
                    if _clock is not None:
                        _mapped = _clock.map_row(int(_r["iter"]), _step)
                        if _mapped is None:
                            _window_skip = True
                        else:
                            _window_it, _step = _mapped
                    if not _window_skip:
                        _window_proposed = self.windowed_stop.observe(
                            _window_it, _r["loss"], _step
                        )
                    if (_window_proposed and self.pose_pre is not None
                            and hasattr(self.windowed_stop,
                                        "note_barred_proposal")):
                        _window_bad = (
                            float(self.pose_pre.anomalous()) > 0.5
                            or float(self.pose_pre.drifted()) > 0.5
                        )
                        if _window_bad:
                            self.windowed_stop.note_barred_proposal()
                            _window_proposed = False
                    if not _window_skip:
                        self.windowed_sweep.observe(
                            _window_it, _r["loss"], _step
                        )
                if _r["loss"] < _best_loss_host:
                    _best_loss_host = _r["loss"]
                    # MUST be .clone(). The drained rows are views into one
                    # host buffer that the next drain overwrites.
                    _best_rot_host = _r["rot"].clone()
                    _best_tran_host = _r["tran"].clone()
                    _best_coverage_host = _coverage
                if (_window_proposed and self.closed_loop_guard.enabled):
                    _candidate_w2c = _w2c_from(
                        _best_rot_host, _best_tran_host
                    ).detach().cpu().numpy()
                    _innovation = _pose_distance_numpy(
                        init_rel_w2c, _candidate_w2c,
                        self._windowed_scales,
                    )
                    _guard_action = self.closed_loop_guard.decide_proposal(
                        _coverage, _innovation,
                        can_stop=(_r["iter"] + 1 < num_iters),
                    )
                    if _guard_action == "veto":
                        if hasattr(self.windowed_stop,
                                   "note_barred_proposal"):
                            self.windowed_stop.note_barred_proposal()
                        _window_proposed = False
                    elif _guard_action == "audit":
                        # Retain the proposal in the convergence rule so its
                        # end_frame() compares it with the true full-tail best.
                        _window_proposed = False
                # ESSignalBuffer's pose_delta is already exactly the quantity
                # the eager path computes for this: the norm of the pose change
                # across the optimizer step, over rot(4)+tran(3).
                _legacy_stop = self.stopper.check(
                    _r["iter"], _r["loss"], _r["pose_delta"]
                )
                if (_legacy_stop
                        or (_window_proposed and self.windowed_stop.active)):
                    return True
            return False

        def _window_phase(iteration):
            """Update regime whose evidence must not share one window."""
            _use_pre_at = (
                self.pose_pre is not None
                and (self.pre_handoff <= 0 or iteration < self.pre_handoff)
            )
            if not _use_pre_at:
                return "adam"
            if (self.pose_pre.diag_after > 0
                    and iteration >= self.pose_pre.diag_after):
                return "pre-diagonal"
            if (self.pose_pre.restart_at > 0
                    and iteration >= self.pose_pre.restart_at):
                return "pre-restarted"
            return "pre-full"

        def _w2c_from(rot, trans):
            """Host-side 4x4 from a (4,) quaternion and a (3,) translation."""
            m = torch.eye(4)
            m[:3, :3] = build_rotation(F.normalize(rot[None]))[0]
            m[:3, 3] = trans
            return m

        # Timed region: the optimisation loop only. Excludes this frame's setup
        # (odometry, render settings, the initial loss check) and all of
        # mapping, so it is the quantity a tracking-side change can actually
        # move.
        _track_t0 = time.perf_counter()

        # NVTX marker so a profiler can isolate TRACKING's rasterizer kernels
        # from MAPPING's - they launch the same kernel at the same resolution
        # and no launch property separates them. Mirrors MonoGS's
        # "monogs_tracking" range so one profiling script covers both models.
        #
        # NOTE: push/pop ranges are THREAD-LOCAL and autograd runs backward on
        # its own CUDA worker thread, so this range does NOT cover
        # renderCUDABackward unless the profiler run also forces backward onto
        # the calling thread. GSLAM_PROFILE_SYNC_AUTOGRAD=1 does that; it is
        # profiling-only and must never be set for a timing run.
        if os.environ.get("GSLAM_PROFILE_SYNC_AUTOGRAD") == "1" and hasattr(
            torch.autograd, "set_multithreading_enabled"
        ):
            torch.autograd.set_multithreading_enabled(False)
        torch.cuda.nvtx.range_push("gslam_tracking")

        _window_phase_label = _window_phase(0)
        # PER-FRAME ADAPTIVE-TRANSITION STATE. The switch is decided fresh
        # every frame - that is the point of it. adaptive_switches in the
        # preconditioner summary reports how many frames actually switched,
        # and _adaptive_switch_iters their distribution, which is what makes
        # the learned transition readable afterwards instead of implicit.
        _ad_enabled = (self.pose_pre is not None
                       and self.pose_pre.adaptive_diag)
        _ad_active = False
        _ad_pending_loss = None
        _ad_pending_q = None
        _ad_pending_t = None
        _ad_failures = 0
        _ad_switch_iteration = None
        # A frame that never switches is RIGHT-CENSORED evidence, not a
        # missing sample: it says no transition was needed before the frame
        # ended. The class wants that distinction, so the flag is captured at
        # frame start rather than inferred afterwards.
        _ad_calibration_frame = (
            _ad_enabled
            and self.pose_pre.adaptive_diag_calibration_frames > 0)
        # TILE-LEVEL SPARSE SAMPLING. Built ONCE per frame and reused across
        # every iteration, exactly as SplaTAM and MonoGS do - the ranking is
        # over the GT image, which does not change within a frame, so
        # rebuilding it per iteration would be pure cost.
        #
        # ALWAYS-SPARSE, NOT A MIDDLE WINDOW. is_sparse_phase divides by the
        # iteration CAP, not the realised count, so a [0.2, 0.8) window on a
        # frame that stops early ends INSIDE the sparse phase and never reaches
        # its closing full-resolution iterations - which is how the MonoGS arm
        # ended up committing poses from masked renders. [0.0, 1.0) sidesteps
        # that entirely and is also the only shape compatible with the
        # iteration graph, which needs the mask constant within a frame.
        # SplaTAM ships exactly this for the same two reasons.
        #
        # THE FINAL ITERATION IS FORCED DENSE. With an always-sparse window
        # every committed pose would otherwise come from a masked render. One
        # full-resolution iteration before the frame ends is what final_dense
        # does on the other two models; here it is unconditional because there
        # is no window to fall back on.
        _ps_mask = None
        if self.pixel_sample_cfg.get("enabled", False):
            _ps_H, _ps_W = gt_depth.shape[:2]
            _ps_mask = build_tile_mask(gt_color, _ps_H, _ps_W,
                                       self.pixel_sample_cfg)
            self._ps_frames += 1

        # 1 / (fraction of the image the mask leaves rendered). Computed once
        # per frame; the mask is constant within a frame, so this is too.
        _ps_scale = None
        if _ps_mask is not None:
            _ps_scale = 1.0 / kept_pixel_fraction(_ps_mask, _ps_H, _ps_W)

        _ps_force_dense = False
        _ps_dense_deadline = None

        def _active_mask_and_scale():
            """(mask, loss_scale) for THIS iteration - (None, None) when dense.

            Returned as a PAIR and read once per iteration, because the two
            must agree: a loss rescaled onto the dense scale while the render
            is masked (or the reverse) is worse than either alone. Reading them
            from two separate calls invited exactly that, since the mask getter
            also has a counter side effect.

            NOT gated on `iter >= num_iters - 1`. The frame usually stops
            early, so the cap's last iteration never runs and that test would
            leave every committed pose coming from a masked render - which is
            exactly the failure the MonoGS sparse arm shipped with. Two paths
            set _ps_force_dense instead: the stop, and the cap.
            """
            if _ps_mask is None or _ps_force_dense:
                return None, None
            self._ps_iters_masked += 1
            return _ps_mask, _ps_scale

        if self.grad_reuse.enabled and _ad_enabled:
            # THE ADAPTIVE DIAGONAL SCORES A PROPOSAL WITH THE NEXT
            # ITERATION'S LOSS: a full step taken at iteration k is judged
            # by whether the render at k+1 improves on L_k. A reuse
            # iteration renders nothing, so it has no L_{k+1} to judge
            # with - the rule would score every proposal against a repeat
            # of its own loss, read it as a failure, and roll back on a
            # cadence set by the reuse period rather than by the data.
            #
            # Refused rather than hooked. The two mechanisms want the same
            # iteration for different things and there is no version of
            # that which measures cleanly.
            raise ValueError(
                "grad_reuse cannot run with the adaptive diagonal: it scores "
                "each proposal with the next iteration's loss, and a reuse "
                "iteration produces none")
        self.stale_probe.reset_frame()
        # The stash NEVER crosses a frame boundary: between frames the
        # pose jumps to a new initialisation, so a carried gradient would
        # have been evaluated at a pose the optimiser is nowhere near.
        self.grad_reuse.reset_frame()
        _gr_stash = [None]
        _gr_last = [None]
        _gr_reused = [False]
        # OUT-CHANNEL FOR _gxi, NOT THE RETURN TUPLE. _tracking_iteration's
        # return shape is shared with _adaptive_iteration's (both unpacked
        # into the same four names below), so extending it would mean
        # touching a closure this feature never uses - and the construction
        # guard above already proves _ad_enabled and grad_reuse.enabled
        # cannot both be true, so that closure could never fill a 5th slot
        # anyway. A plain list assignment is not a CUDA op, so it is exactly
        # as safe to do inside a captured closure as the reuse bookkeeping
        # above already is: written once during capture, and because the
        # kernels PRODUCING _gxi are themselves part of the replayed
        # sequence, _gxi_out[0] reflects the latest replay's value with no
        # Python re-execution needed - the same trick the class's own
        # docstring describes for the tuple it returns.
        _gxi_out = [None]
        _gr_pose_params = [opt_cam_rot, opt_cam_trans]
        # THE TRACE RECORDS THE ABSOLUTE w2c, NOT THE OPTIMISED VARIABLE.
        # Gaussian-SLAM optimises L in W = A L (A = reference_w2c, fixed for
        # the frame). Rotation distances are the same either way, but the
        # camera centre of L is not the camera's, so the plot's mm figures
        # would be wrong. The gradient is carried into W's left tangent too,
        # g_abs = Ad_{A^-1}^T g_rel, so both models' cosines share a basis.
        _it_A = _it_map = None
        if self.iter_trace.wants(frame_id):
            _it_A = torch.from_numpy(reference_w2c).to(
                device="cuda", dtype=torch.float64)
            _it_map = se3_adjoint(torch.linalg.inv(_it_A)).T

        def _it_abs(rel):
            """Absolute (q, t) from a relative 4x4 w2c, or from (q, t)."""
            if isinstance(rel, tuple):
                q, t = rel
                m = torch.eye(4, dtype=torch.float64, device="cuda")
                m[:3, :3] = build_rotation(
                    F.normalize(q.detach().reshape(1, 4)))[0].double()
                m[:3, 3] = t.detach().reshape(3).double()
                rel = m
            W = _it_A @ rel.to(device="cuda", dtype=torch.float64)
            return mat_to_quat_capturable(W[:3, :3]), W[:3, 3]

        if _it_A is not None:
            self.iter_trace.begin_frame(
                frame_id, num_iters, *_it_abs((opt_cam_rot, opt_cam_trans)))
        for iter in range(num_iters):
            # SECOND FORCED-DENSE TRIGGER: THE CAP. The stop path below grants
            # a dense iteration when the frame converges, but a frame that runs
            # its full budget never reaches that path, and every candidate it
            # ever scored was masked. Arming it on the last iteration means the
            # cap commits densely too. Before this, a run whose stopper was
            # vetoed committed 462/590 poses from masked renders.
            if (_ps_mask is not None and not _ps_force_dense
                    and iter == num_iters - 1):
                _ps_force_dense = True
                self._ps_dense_commits += 1
            _bin_kwargs = self.binning_capacity.render_kwargs()
            # Before the handoff the preconditioner drives; after it, Adam.
            _use_pre = (self.pose_pre is not None
                        and (self.pre_handoff <= 0 or iter < self.pre_handoff))

            if self.stale_probe.wants(frame_id, iter):
                # SAFE TO BACKWARD HERE: this model zeroes .grad at the END
                # of each iteration, so at the top of one there is nothing
                # to accumulate onto and the probe reads its own gradient
                # rather than a sum. observe() also saves and restores
                # .grad around the call - without that, the probe gradient
                # is added to the next real one and the only symptom is a
                # worse ATE.
                def _gs_backward(_unused=None, _bk=_bin_kwargs):
                    _c, _d, _, _, _ = self.compute_losses(
                        gaussian_model, render_settings,
                        opt_cam_rot, opt_cam_trans,
                        gt_color, gt_depth, depth_mask,
                        binning_kwargs=_bk)
                    (self.w_color_loss * _c
                     + (1 - self.w_color_loss) * _d).backward()

                def _gs_grad():
                    # THE 6-D SE(3) TANGENT, not the raw 7-vector. The
                    # quaternion carries a gauge direction that moves no
                    # pose at all; leaving it in drags the cosine down
                    # while meaning nothing, so a 7-D number UNDERSTATES
                    # how well the pose gradient survives. Same analytic
                    # map the preconditioner steps with.
                    if (opt_cam_rot.grad is None
                            or opt_cam_trans.grad is None):
                        return None
                    with torch.no_grad():
                        return quat_trans_grad_to_tangent(
                            opt_cam_rot.detach().reshape(4),
                            opt_cam_trans.detach().reshape(3),
                            opt_cam_rot.grad.detach().reshape(4),
                            opt_cam_trans.grad.detach().reshape(3),
                        ).reshape(-1)

                self.stale_probe.observe(
                    frame_id, iter, [opt_cam_rot, opt_cam_trans],
                    _gs_backward, grad_map_fn=_gs_grad)

            # GRADIENT REUSE. The whole render, loss and backward are
            # skipped - not one stage of them - so preprocess, binning
            # and the sort go with them.
            #
            # NEVER ON THE LAST ITERATION. This model FORCES A DENSE
            # render there (the cap trigger above) precisely so the pose
            # commits from a real render rather than a masked one. A
            # reuse iteration renders nothing at all, so reusing there
            # would defeat the guard that fixed 462/590 poses committing
            # from masked renders.
            #
            # DECIDED HERE, NOT INSIDE THE CLOSURE, because the CALLER
            # needs the answer: a reuse iteration has to bypass the CUDA
            # graph, and iter_graph.run() must be told before it would
            # otherwise replay. The closure only READS the flag now.
            # restore_grads() is a copy_ into the existing .grad, never a
            # rebind, so the addresses a capture recorded stay valid.
            _gr_reused[0] = (
                self.grad_reuse.should_reuse(iter, iter >= num_iters - 1)
                and restore_grads(_gr_pose_params, _gr_stash[0]))
            # COUNTED HERE FOR THE SAME REASON, and this one is a
            # REPORTING bug if it is left inside. A rendered iteration
            # REPLAYS under the graph - the Python body does not run - so
            # note_rendered() inside the closure would only ever count the
            # warmup and the capture, and the summary would read ~5000
            # reused against ~5 rendered. Both counters live on the eager
            # side of the call so the line stays true with the graph on.
            # DROP THE PREVIOUS ITERATION BEFORE THE CAPTURE RUNS.
            # _gr_last holds iteration N-1's loss tensors, which keeps its
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
            if _clock is not None:
                _clock.note(bool(_gr_reused[0]))
            if _gr_reused[0]:
                self.grad_reuse.note_reused()
            elif self.grad_reuse.enabled:
                # Gated so the counters stay at zero, and the disabled path
                # stays free, when reuse is off.
                self.grad_reuse.note_rendered()
            # Render, loss, backward and optimizer step as one callable, so
            # TrackingIterationGraph can record the whole thing into a CUDA
            # graph. Everything that synchronises stays outside it, below.
            def _tracking_iteration(_bk=_bin_kwargs, _it=iter):
                # HOISTED ABOVE THE BRANCH. _ps_s is read much further
                # down this function - outside the if/else - to rescale
                # the coverage the closed-loop guard consumes. Binding it
                # only on the render path left it unbound on every reuse
                # iteration. It is cheap: the mask itself is built once
                # per frame, this only selects it.
                _ps_m, _ps_s = _active_mask_and_scale()
                if _gr_reused[0]:
                    # ALL FOUR NAMES THE CLOSURE RETURNS, not just the
                    # total. _color_loss and _depth_loss are returned too
                    # and would be unbound here - the same
                    # UnboundLocalError the MonoGS port hit on _pkg.
                    (_total_loss, _color_loss, _depth_loss,
                     _alpha_mask) = _gr_last[0]
                else:
                    _color_loss, _depth_loss, _, _, _alpha_mask = self.compute_losses(
                        gaussian_model, render_settings, opt_cam_rot, opt_cam_trans,
                        gt_color, gt_depth, depth_mask, binning_kwargs=_bk,
                        tile_mask=_ps_m, loss_scale=_ps_s)
                    _total_loss = (self.w_color_loss * _color_loss
                                   + (1 - self.w_color_loss) * _depth_loss)
                    _total_loss.backward()
                    # BETWEEN backward() and the step: zero_grad below
                    # clears .grad in place, so after it there is nothing
                    # left to stash.
                    if self.iter_trace.active:
                        self.iter_trace.capture(
                            opt_cam_rot, opt_cam_trans,
                            opt_cam_rot.grad, opt_cam_trans.grad,
                            tangent_map=_it_map)
                    if self.grad_reuse.enabled:
                        _gr_stash[0] = stash_grads(_gr_pose_params)
                        _gr_last[0] = (_total_loss, _color_loss,
                                       _depth_loss, _alpha_mask)
                self.es_signals.record_pre_step(opt_cam_rot, opt_cam_trans)
                # Pose snapshot for the early-stop pose delta. Taken here,
                # before the step, and differenced after it - the same
                # definition exp/adaptive_mapping used (prev_rot/prev_trans).
                # cat() copies, so this is a genuine snapshot rather than an
                # alias that the step would overwrite.
                _prev_pose = torch.cat([opt_cam_rot.detach().reshape(-1),
                                        opt_cam_trans.detach().reshape(-1)])
                # THE 7-VECTOR GRADIENT, MAPPED TO THE SE(3) TANGENT.
                # opt_cam_rot is an UNNORMALISED quaternion that the render
                # path F.normalize's, so it carries a gauge direction that is
                # an exact null direction of the loss. dq/dtheta is orthogonal
                # to q by construction, so the contraction discards that
                # component automatically - which is the whole Stage 0 problem
                # on SplaTAM, gone by construction rather than by projection.
                _gxi = None
                _q0 = _t0 = None
                if self.pose_pre is not None and opt_cam_rot.grad is not None:
                    # SNAPSHOT THE POSE BEFORE ADAM TOUCHES IT.
                    #
                    # THE BUG THIS FIXES, and it invalidated every GSLAM number
                    # measured before it. apply_tangent_step composes
                    # T <- exp(dxi) T, so it needs the pose the gradient was
                    # evaluated at. Reading opt_cam_rot AFTER
                    # optimizer.step() meant composing the preconditioned step
                    # ON TOP OF Adam's - the pose moved TWICE per iteration,
                    # once by each rule. The comment here used to claim Adam's
                    # step was "overwritten", which is what MonoGS does
                    # (copy_ of the delta, so Adam's value is discarded) and
                    # what SplaTAM does (its preconditioned path never calls
                    # optimizer.step() at all). Gaussian-SLAM composes from the
                    # live tensor, so it needed the snapshot.
                    #
                    # clone() because optimizer.step() modifies in place.
                    _q0 = opt_cam_rot.detach().clone().reshape(4)
                    _t0 = opt_cam_trans.detach().clone().reshape(3)
                    _gxi = quat_trans_grad_to_tangent(
                        _q0, _t0,
                        opt_cam_rot.grad.detach().reshape(4),
                        opt_cam_trans.grad.detach().reshape(3))
                # HAND _gxi OUT THROUGH THE SIDE CHANNEL, NOT A DIRECT CALL
                # HERE. This closure gets CAPTURED into a CUDA graph once
                # warmup ends, and every iteration after that REPLAYS -
                # replay runs no Python at all, so a call placed here (as an
                # earlier version of this patch did) would execute on the
                # capture iteration and never again for the rest of the
                # frame. That bug produced exactly the observed symptom: one
                # trust check per frame, a boundary frozen at its initial
                # value forever, and it would ALSO have risked a host sync
                # landing inside the capture itself on whichever iteration
                # happened to be the check boundary - undefined at best.
                # A plain list write is not a CUDA op, so it is safe under
                # capture; note_fresh_grad() itself (with its real .item()
                # sync) is called from the genuinely eager block after
                # iter_graph.run() returns, below.
                _gxi_out[0] = _gxi
                # Adam still steps - the optimiser also holds the Gaussian
                # params (at lr 0.0, so they do not move). Its POSE step is
                # DISCARDED below when the preconditioner is driving, because
                # the tangent step is composed from the pre-step snapshot.
                gaussian_model.optimizer.step()
                if _gxi is not None:
                    if _use_pre:
                        with torch.no_grad():
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
                            # GRAD_REUSE_STEP_SCALE: damp the APPLIED step on
                            # a reuse iteration only, after the trust-region
                            # clamp already ran inside step() - see
                            # utils/grad_reuse.py's step_scale docstring.
                            # Never touches m/M/P, so it composes cleanly
                            # with no_accum/freeze_m above. Baked to a no-op
                            # for a captured RENDER iteration: _gr_reused[0]
                            # is always False when this body gets captured.
                            if (_gr_reused[0]
                                    and self.grad_reuse.step_scale != 1.0):
                                _dxi = _dxi * self.grad_reuse.step_scale
                            _qn, _tn = apply_tangent_step(
                                _q0, _t0, _dxi, mat_to_quat_capturable)
                            # exp() preserves the norm exactly, so q stays unit
                            # by construction rather than by renormalisation.
                            self.pose_pre.note_gauge(_qn.norm())
                            opt_cam_rot.copy_(_qn.view_as(opt_cam_rot))
                            opt_cam_trans.copy_(_tn.view_as(opt_cam_trans))
                    else:
                        # ADAM PHASE OF A HANDOFF. Keep feeding the plateau
                        # tracker: step() is not being called, so _g_best and
                        # the stall counter would FREEZE at the handoff
                        # iteration and a frame that had not already stalled
                        # could never stop.
                        with torch.no_grad():
                            self.pose_pre.observe(_gxi)
                # set_to_none=False is REQUIRED under capture, and this was
                # set_to_none=True. With set_to_none=True the .grad tensors are
                # freed and reallocated every iteration, so the addresses the
                # captured kernels were recorded against stop being the
                # addresses autograd actually writes to.
                gaussian_model.optimizer.zero_grad(set_to_none=False)
                _guard_extra = None
                if self.closed_loop_guard.enabled:
                    # Coverage belongs to the same pre-step pose as the loss.
                    # The scalar reduction remains on-device and is copied only
                    # with the existing batched early-stop drain.
                    # COVERAGE MUST BE MASK-AWARE. alpha is 0 in masked tiles,
                    # so a raw mean over the whole image reports ~sample_ratio
                    # times the true coverage purely by construction. The
                    # guard's history, its coverage veto and its
                    # critical-render-collapse latch all consume this number,
                    # and feeding it a mix of masked and dense frames is what
                    # latched the guard and killed early stopping for an entire
                    # run. Same 1/kept_fraction scale as the loss.
                    _cov = _alpha_mask.to(torch.float32).mean()
                    if _ps_s is not None:
                        _cov = _cov * _ps_s
                    _guard_extra = [_cov]
                self.es_signals.record_post_step(
                    _total_loss, opt_cam_rot, opt_cam_trans,
                    extra=_guard_extra,
                )
                _now_pose = torch.cat([opt_cam_rot.detach().reshape(-1),
                                       opt_cam_trans.detach().reshape(-1)])
                # Returned as a DEVICE tensor. Reading it is a sync, legal
                # outside the capture but not inside, so whether to read it at
                # all is the caller's decision - the batched path never does.
                _pose_delta = (_now_pose - _prev_pose).norm()
                return _total_loss, _color_loss, _depth_loss, _pose_delta

            # Release the previous iteration's autograd graph before the
            # capture iteration runs, or the capture dies in backward() with
            # "operation would make the legacy stream depend on a capturing
            # blocking stream".
            #
            # Autograd stamps each Node with the stream current at FORWARD
            # time and runs it on that stream during backward. AccumulateGrad
            # is cached per leaf via a weak_ptr, so any live reference to a
            # previous loss keeps the eager-warmup accumulators - stamped with
            # the legacy default stream - alive to be reused inside the
            # capture. Dropping the references lets the weak_ptrs expire so
            # they are rebuilt on the capture stream.
            total_loss = color_loss = depth_loss = pose_delta = None

            # PHASE tells the graph which KIND of iteration this is. With a
            # handoff the frame has two: the preconditioned update and Adam's.
            # Each gets its own capture, so a graph never replays the wrong
            # one. Costs a second instantiate and warmup per frame - see
            # TrackingIterationGraph.run.
            def _adaptive_iteration(_bk=_bin_kwargs):
                """The acquisition-phase iteration, with the decision inline.

                Identical to _tracking_iteration except that the render and
                backward are separated from the UPDATE, so the frame's own loss
                can score the previous full-matrix proposal before this
                iteration commits anything. That split is why this path cannot
                be captured, and why it exists only for this arm - every other
                run still takes the combined closure above byte for byte.

                The proposal/score pairing: a full step taken at iteration k
                from pose P_k with loss L_k is a PROPOSAL. Iteration k+1
                renders at the resulting pose and produces L_{k+1}. If
                L_{k+1} does not improve on L_k, that is one failure. After
                `patience` consecutive failures the pose is restored to P_k and
                the diagonal step - computed from the SAME retained gradient
                and metric that produced the rejected proposal - replaces it.
                """
                nonlocal _ad_active, _ad_pending_loss, _ad_pending_q
                nonlocal _ad_pending_t, _ad_failures, _ad_switch_iteration
                _ps_m, _ps_s = _active_mask_and_scale()
                _color_loss, _depth_loss, _, _, _alpha_mask = self.compute_losses(
                    gaussian_model, render_settings, opt_cam_rot, opt_cam_trans,
                    gt_color, gt_depth, depth_mask, binning_kwargs=_bk,
                    tile_mask=_ps_m, loss_scale=_ps_s)
                _total_loss = (self.w_color_loss * _color_loss
                               + (1 - self.w_color_loss) * _depth_loss)
                _total_loss.backward()
                self.es_signals.record_pre_step(opt_cam_rot, opt_cam_trans)
                _prev_pose = torch.cat([opt_cam_rot.detach().reshape(-1),
                                        opt_cam_trans.detach().reshape(-1)])

                # THE ONE SYNC THIS ARM INTRODUCES, and it is bounded: it stops
                # the moment the frame latches to diagonal mode.
                _reject = False
                if not _ad_active:
                    _cur = float(_total_loss.detach())
                    if _ad_pending_loss is not None:
                        _improved = adaptive_loss_improved(_ad_pending_loss, _cur)
                        _ad_failures, _reject = adaptive_failure_streak(
                            _ad_failures, _improved,
                            self.pose_pre.adaptive_diag_patience)
                        # activate_adaptive_diagonal raises without a metric,
                        # and on the very first frame of a CARRY=0 run the
                        # streak can reach patience before one exists. Hold
                        # the rejection rather than crash: the next iteration
                        # re-tests it against a metric that now exists.
                        if _reject and not self.pose_pre._have_metric:
                            _reject = False

                if _reject:
                    with torch.no_grad():
                        # Restore the pose the rejected proposal was taken
                        # FROM, then let the class replace that proposal.
                        opt_cam_rot.copy_(_ad_pending_q.view_as(opt_cam_rot))
                        opt_cam_trans.copy_(_ad_pending_t.view_as(opt_cam_trans))
                        _dxi = self.pose_pre.activate_adaptive_diagonal()
                        _qn, _tn = apply_tangent_step(
                            _ad_pending_q, _ad_pending_t, _dxi,
                            mat_to_quat_capturable)
                        self.pose_pre.note_gauge(_qn.norm())
                        opt_cam_rot.copy_(_qn.view_as(opt_cam_rot))
                        opt_cam_trans.copy_(_tn.view_as(opt_cam_trans))
                    _ad_active = True
                    _ad_switch_iteration = iter + 1
                    _ad_pending_loss = None
                    _ad_pending_q = _ad_pending_t = None
                    _ad_failures = 0
                    # This iteration's own gradient is deliberately unused: the
                    # diagonal step REPLACED the rejected proposal rather than
                    # following it. Adam still steps for the Gaussian params
                    # (lr 0.0), and the grads are cleared as usual below.
                    gaussian_model.optimizer.step()
                else:
                    _q0 = opt_cam_rot.detach().clone().reshape(4)
                    _t0 = opt_cam_trans.detach().clone().reshape(3)
                    _gxi = None
                    if opt_cam_rot.grad is not None:
                        _gxi = quat_trans_grad_to_tangent(
                            _q0, _t0,
                            opt_cam_rot.grad.detach().reshape(4),
                            opt_cam_trans.grad.detach().reshape(3))
                    gaussian_model.optimizer.step()
                    if _gxi is not None:
                        with torch.no_grad():
                            _dxi = self.pose_pre.step(_gxi)
                            _qn, _tn = apply_tangent_step(
                                _q0, _t0, _dxi, mat_to_quat_capturable)
                            self.pose_pre.note_gauge(_qn.norm())
                            opt_cam_rot.copy_(_qn.view_as(opt_cam_rot))
                            opt_cam_trans.copy_(_tn.view_as(opt_cam_trans))
                        if not _ad_active:
                            # Score THIS step at the next render.
                            _ad_pending_loss = _cur
                            _ad_pending_q = _q0
                            _ad_pending_t = _t0

                gaussian_model.optimizer.zero_grad(set_to_none=False)
                _guard_extra = None
                if self.closed_loop_guard.enabled:
                    # COVERAGE MUST BE MASK-AWARE. alpha is 0 in masked tiles,
                    # so a raw mean over the whole image reports ~sample_ratio
                    # times the true coverage purely by construction. The
                    # guard's history, its coverage veto and its
                    # critical-render-collapse latch all consume this number,
                    # and feeding it a mix of masked and dense frames is what
                    # latched the guard and killed early stopping for an entire
                    # run. Same 1/kept_fraction scale as the loss.
                    _cov = _alpha_mask.to(torch.float32).mean()
                    if _ps_s is not None:
                        _cov = _cov * _ps_s
                    _guard_extra = [_cov]
                self.es_signals.record_post_step(
                    _total_loss, opt_cam_rot, opt_cam_trans,
                    extra=_guard_extra,
                )
                _now_pose = torch.cat([opt_cam_rot.detach().reshape(-1),
                                       opt_cam_trans.detach().reshape(-1)])
                return (_total_loss, _color_loss, _depth_loss,
                        (_now_pose - _prev_pose).norm())

            if _ad_enabled:
                total_loss, color_loss, depth_loss, pose_delta =                     _adaptive_iteration()
            else:
                total_loss, color_loss, depth_loss, pose_delta = self.iter_graph.run(
                # PHASE 1 ONLY FOR THE ADAM SIDE OF A REAL HANDOFF. Keying
                # it on `_use_pre` alone was wrong: with no preconditioner
                # _use_pre is False, so phase stayed 1 while reset_for_frame
                # reset _phase to 0 - a spurious switch on the first iteration
                # of EVERY frame. Harmless (the graph is already None there, so
                # captures and replays were unaffected) but it reported
                # "590 phase switches (1.00/frame)" on a run with no
                # preconditioner at all, which is exactly the kind of log line
                # that gets believed later.
                    _tracking_iteration,
                    bypass=_gr_reused[0],
                    phase=(1 if (self.pose_pre is not None
                                 and self.pre_handoff > 0
                                 and not _use_pre) else 0)
                )
            # ADAPTIVE-COOLDOWN TRUST SIGNAL, HERE AND ONLY HERE. This block
            # runs EVERY iteration regardless of graph/replay state - unlike
            # the closure above, which stops executing Python entirely once
            # a capture exists (see the side-channel comment at _gxi_out's
            # assignment for what went wrong when this lived inside it).
            # Only on a RENDER iteration: on a reuse one, .grad was populated
            # by restore_grads(), so _gxi_out[0] would be a copy of the last
            # real gradient, and comparing it against itself would report a
            # meaningless cos=1.0 every time. Safe to call unconditionally
            # with a None _gxi (no preconditioner, or no grad yet) -
            # note_fresh_grad() treats that as no evidence, not distrust.
            if not _gr_reused[0]:
                self.grad_reuse.note_fresh_grad(_gxi_out[0], iter)
            if self.iter_trace.active:
                # total_loss is the PRE-step render's loss, the pose after
                # this iteration's step - the same pairing SplaTAM records.
                self.iter_trace.note_iteration(
                    iter, total_loss, *_it_abs((opt_cam_rot, opt_cam_trans)),
                    reused=_gr_reused[0])
            if _use_pre:
                # REBUILD P. The eigendecomposition is not capturable and does
                # not belong inside the iteration. Without this call P stays at
                # the identity and every step takes the no-metric fallback,
                # which looks like a working run and measures nothing.
                #
                # SKIPPED DURING THE ADAM PHASE: refactor() is also where the
                # step counter advances, so counting it on iterations that never
                # called step() dilutes every step statistic - it read 0.02x
                # Adam where the truth was 0.21x on SplaTAM.
                self.pose_pre.refactor()
            self.binning_capacity.note_iteration()
            _iters_run = iter + 1
            _stop_now = False

            if self.es_signals.enabled:
                # Nothing is read back here - the drain below syncs once every
                # `batch` iterations instead of every iteration.
                self.es_signals.note_iteration()
                _stop_now = _consume(self.es_signals.drain_if_full())
                if self.windowed_stop.enabled and iter + 1 < num_iters:
                    _next_phase = _window_phase(iter + 1)
                    if _next_phase != _window_phase_label:
                        # A phase boundary can land halfway through batch=8.
                        # Consume the old phase's partial ring before resetting
                        # both statistical windows, or rows on either side of
                        # the boundary would be treated as one regime.
                        _stop_now = (
                            _consume(self.es_signals.drain_phase()) or _stop_now
                        )
                        _phase_it = (_clock.before(iter + 1)
                                     if _clock is not None else iter + 1)
                        self.windowed_stop.start_phase(_phase_it, _next_phase)
                        self.windowed_sweep.start_phase(_phase_it, _next_phase)
                        _window_phase_label = _next_phase
            else:
                with torch.no_grad():
                    _loss_host = total_loss.item()
                    if _loss_host < current_min_loss:
                        current_min_loss = _loss_host
                        _best_loss_host = _loss_host
                        _best_rot_host = F.normalize(opt_cam_rot[None].clone().detach().cpu())[0]
                        _best_tran_host = opt_cam_trans.clone().detach().cpu()
                    _stop_now = self.stopper.check(iter, _loss_host, pose_delta.item())

            # PLATEAU STOPPING: a patience on the BEST |g| this frame, not a
            # threshold on the current one. |g|/|g_0| is non-monotone and
            # plateaus, so a threshold either fires on a noise dip or waits far
            # too long. stalled() is gated on the frame having beaten its own
            # starting gradient, so a DIVERGING frame - which also stops
            # improving - runs its full budget rather than quitting at the
            # floor. Both measured on SplaTAM.
            if (not _stop_now and self.pose_pre is not None
                    and self.pre_stop_rel > 0.0
                    and self.pre_stop_mode == "best"
                    and iter >= self.pre_stop_min
                    and iter % self.pre_stop_every == 0):
                if float(self.pose_pre.stalled()) >= self.pre_stop_patience:
                    _stop_now = True
            # THE GRANTED DENSE ITERATION HAS NOW RUN, so commit.
            #
            # Without this the grant below cancelled the stop and nothing ever
            # re-armed it, so the frame carried on DENSE TO THE CAP: ~95 extra
            # iterations per stopping frame (12168 dense iterations across 128
            # granting frames, measured), each one handing the guard a dense
            # coverage reading while its history was otherwise being filled
            # from masked frames. That skewed the median upward until ordinary
            # masked frames read as collapsed, which latched the guard.
            if _ps_dense_deadline is not None and iter >= _ps_dense_deadline:
                break
            if _stop_now and _ps_mask is not None and not _ps_force_dense:
                # ONE FULL-RESOLUTION ITERATION BEFORE COMMITTING.
                #
                # The committed pose is the best-loss candidate, and with an
                # always-sparse window every candidate so far was scored on a
                # MASKED render. Granting one dense iteration puts an unmasked
                # candidate in the running before the frame ends. This is what
                # final_dense does on SplaTAM and MonoGS; on SplaTAM its
                # absence cost ATE 7.96 against 4.04.
                # EXACTLY ONE: the deadline above ends the frame on the next
                # iteration. And it only puts a genuine candidate in the
                # running because compute_losses now rescales the masked loss
                # onto the dense scale - otherwise the dense candidate's larger
                # pixel sum loses the best-loss comparison by construction, and
                # this whole path buys nothing.
                _ps_force_dense = True
                _ps_dense_deadline = iter + 1
                _stop_now = False
                self._ps_dense_commits += 1
            if _stop_now:
                # Early stopping fired. The committed pose is the best-loss
                # candidate, not the current one, so stopping here cannot
                # commit a half-converged pose - the remaining iterations could
                # only have added more candidates.
                break

            # The diagnostic pose is computed ONLY on the iterations that
            # actually log it.
            #
            # It used to be computed every iteration and printed on one in
            # twenty. That cost four host round-trips per iteration - two .cpu()
            # copies, plus assigning CUDA tensors into a CPU torch.eye(4), which
            # is an implicit device-to-host copy - none of which feeds the
            # optimizer or the committed pose. Only best_w2c does, and that is
            # handled above. The last iteration's log moved below the loop
            # because under batching the best candidate is not final until the
            # remainder has been drained.
            if iter % 20 == 0 and iter != num_iters - 1:
                with torch.no_grad():
                    cur_quat = F.normalize(opt_cam_rot[None].clone().detach())
                    cur_trans = opt_cam_trans.clone().detach()
                    cur_rel_w2c = torch.eye(4)
                    cur_rel_w2c[:3, :3] = build_rotation(cur_quat)[0]
                    cur_rel_w2c[:3, 3] = cur_trans
                    cur_c2w = torch.inverse(torch.from_numpy(reference_w2c) @ cur_rel_w2c)
                    cur_cam = transformation_to_quaternion(cur_c2w)
                    if (gt_quat * cur_cam[:4]).sum() < 0:  # for logging purpose
                        gt_quat *= -1
                    self.logger.log_tracking_iteration(
                        frame_id, cur_cam, gt_quat, gt_trans, total_loss, color_loss, depth_loss,
                        iter, num_iters, wandb_output=False, print_output=True)

        # Closes "gslam_tracking" - see the push above the loop. Before the
        # drain, so the range covers the optimisation kernels and not the
        # host-side commit.
        torch.cuda.nvtx.range_pop()

        # Drain whatever is still pending, then commit. One sync.
        _consume(self.es_signals.drain_remainder())
        self.windowed_stop.end_frame()
        self.windowed_sweep.end_frame()
        _audit_motion = _audit_loss = None
        if self.closed_loop_guard.enabled and self.closed_loop_guard.frame_audit:
            _window_metrics = self.windowed_stop.metrics(num_iters)
            if (_window_metrics.get("post_motion")
                    and _window_metrics.get("post_loss_gain")):
                _audit_motion = _window_metrics["post_motion"][-1]
                _audit_loss = _window_metrics["post_loss_gain"][-1]
        # Closed here, after the final drain: with es_signals on, the loop
        # above issues work without waiting for it, so stopping the clock
        # before the drain would bill the graph arm for work the GPU had not
        # finished. The drain is the point at which this frame's tracking is
        # genuinely complete.
        torch.cuda.synchronize()
        _frame_track_s = time.perf_counter() - _track_t0
        self._tracking_time_total += _frame_track_s
        if _ad_calibration_frame:
            # ONE SAMPLE PER CALIBRATION FRAME. A frame that switched
            # contributes the iteration it switched at; one that never did
            # contributes its completed iteration count as right-censored
            # evidence. When the Nth sample lands the class installs the
            # median as diag_after, clears adaptive_diag, and every later
            # frame goes back through the combined capturable iteration with
            # no loss readback at all.
            _learned = self.pose_pre.finish_adaptive_calibration_frame(
                _ad_switch_iteration, _iters_run)
            if _learned is not None:
                print(f"[Preconditioner] adaptive calibration complete: "
                      f"fixed restart_at=diag_after={_learned} learned from "
                      f"{self.pose_pre.adaptive_diag_calibration_frames} "
                      f"frames. Later frames drop the per-iteration loss "
                      f"readback.", flush=True)
        # ACTUAL iterations run, not num_iters - with early stopping the loop
        # breaks early, and billing the budget instead of the work would make
        # ms/iter read low by exactly the factor early stopping is saving,
        # hiding the effect inside the metric meant to measure it.
        self._tracking_iters_total += _iters_run
        self._tracking_frames += 1
        self.grad_reuse.note_frame_stop(
            _iters_run,
            "cap" if _iters_run >= num_iters else "stopping",
        )
        # PER-FRAME COUNTS, BECAUSE THE MEAN HIDES A BIMODAL DISTRIBUTION.
        # guarded40 reported 178.1 iters/frame against a budget of 200, which
        # reads like every frame stopping a little early. It was 433 frames at
        # the full 200 and 157 stopping near 118 - a criterion that fires on a
        # quarter of frames, not a little on all of them. Those two have very
        # different implications for whether the rule is worth its cost, and
        # nothing in the summary could tell them apart.
        self._iters_per_frame.append(int(_iters_run))
        self._budget_per_frame.append(int(num_iters))
        self._track_ms_per_frame.append(1000.0 * _frame_track_s)
        # One host read per frame instead of one per iteration. An overflow
        # means duplicateWithKeys dropped instances, so that frame's render was
        # wrong - reported loudly rather than silently absorbed.
        self.binning_capacity.end_frame()

        best_w2c = _w2c_from(_best_rot_host, _best_tran_host)
        if self.closed_loop_guard.enabled:
            _final_innovation = _pose_distance_numpy(
                init_rel_w2c, best_w2c.detach().cpu().numpy(),
                self._windowed_scales,
            )
            if self.closed_loop_guard.should_recover_initial(
                    _initial_coverage_host, _best_coverage_host):
                # The loss can become exactly zero when every rendered pixel
                # disappears behind the tracking mask. Such a pose wins the
                # ordinary argmin despite carrying no information. Recover the
                # frame's original motion prediction before mapping/keyframes
                # see it; their own logic remains completely unchanged.
                best_w2c = torch.from_numpy(init_rel_w2c).to(
                    dtype=best_w2c.dtype
                )
                _best_coverage_host = _initial_coverage_host
                _final_innovation = 0.0
            self.closed_loop_guard.end_frame(
                full_budget=(_iters_run >= num_iters),
                coverage=_best_coverage_host,
                innovation=_final_innovation,
                audit_motion=_audit_motion,
                audit_loss=_audit_loss,
            )
        # Save L_t for the next frame's exact relative-tangent transport. This
        # is the COMMITTED relative pose, not the last iterate.
        self._previous_rel_w2c = best_w2c.detach().clone()
        if self.iter_trace.active:
            # After the closed-loop guard's possible recovery, so the trace's
            # reference is the pose actually committed.
            self.iter_trace.end_frame(*_it_abs(best_w2c.detach()))

        with torch.no_grad():
            # The final iteration's log, moved out of the loop: it reports the
            # COMMITTED pose (reference_w2c @ best_w2c), which is only known
            # once the last batch has been drained.
            #
            # color_loss/depth_loss/total_loss still name the last iteration's
            # tensors. Under the graph those are references into the graph's
            # private pool, and the pool is not rewritten until the next frame's
            # capture - so reading them here is current and correct, but it must
            # happen before reset_for_frame(), not after.
            cur_c2w = torch.inverse(torch.from_numpy(reference_w2c) @ best_w2c)
            cur_cam = transformation_to_quaternion(cur_c2w)
            if (gt_quat * cur_cam[:4]).sum() < 0:  # for logging purpose
                gt_quat *= -1
            self.frame_color_loss.append(color_loss.item())
            self.frame_depth_loss.append(depth_loss.item())
            if self.pose_pre is not None:
                # SAME quantity as logger.log_tracking_iteration's cam_trans_err
                # (mean absolute translation error against ground truth),
                # computed once more here rather than threaded back out of the
                # logger, to keep that call's signature untouched. Cheap and
                # already inside a no_grad, post-sync block - .item() here adds
                # no sync this frame does not already pay (see the
                # torch.cuda.synchronize() above, before the drain).
                #
                # A DELTA, NOT THE LEVEL. Measured (2026-09-22, GSLAM/fr1_desk,
                # a known KEEP scene): the raw level gave a confident, twice-
                # replicated WRONG REDUCE. The level is a diluted, pair-order-
                # confounded estimator of the true effect under alternating
                # interleaving - the delta recovers the full effect at the
                # same frame cost. See utils/online_lr_tuner.py's PoseErrDelta
                # and its POSE_ERR docstring section for the exact mechanism.
                #
                # gt_pose_valid GUARDS THE SCANNET NaN CASE, at BOTH ends: an
                # invalid ground truth pose makes this frame's own level
                # unusable (update() returns None and leaves the delta
                # tracker's reference where it was, so the NEXT valid frame's
                # delta correctly spans back across the gap), and NaN is not
                # None - it would otherwise sail past the tuner's "is there a
                # reading" check and silently poison the whole rung's mean.
                self.pose_pre.note_frame_outcome(
                    pose_err=self._pose_err_delta.update(
                        torch.abs(cur_cam[4:] - gt_trans).mean().item(),
                        valid=gt_pose_valid),
                    at_cap=bool(_iters_run >= num_iters))
            self.logger.log_tracking_iteration(
                frame_id, cur_cam, gt_quat, gt_trans, total_loss, color_loss, depth_loss,
                _iters_run - 1, num_iters, wandb_output=True, print_output=True)

        final_c2w = torch.inverse(torch.from_numpy(reference_w2c) @ best_w2c)
        final_c2w[-1, :] = torch.tensor([0., 0., 0., 1.], dtype=final_c2w.dtype, device=final_c2w.device)
        return torch2np(final_c2w)
