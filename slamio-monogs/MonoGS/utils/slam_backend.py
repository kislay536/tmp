import os
import random
import sys
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.camera_utils import set_capture_safe_pose
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.adaptive_mapper import AdaptiveMapper, autoscale_iteration_bounds


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        # MAPPING-SIDE TIME/ITERATION/FRAME TOTALS for experiments/'s
        # mapping_ms_per_frame / mapping_ms_per_iter / mapping_iters_per_frame
        # columns - nothing in this file aggregated them before. NOT reset in
        # reset() (unlike iteration_count, which schedules gaussian_update_every
        # etc. and is meant to restart): this is a whole-run total.
        self._mapping_time_total = 0.0
        self._mapping_iters_total = 0
        self._mlsys_map_calls = 0
        self._mapping_frames_total = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        # The backend must take the same pose path as the frontend.
        #
        # slam.py calls mp.set_start_method("spawn"), so when the backend runs
        # as a separate process it re-imports camera_utils fresh and the
        # module-level flag the frontend set does NOT carry over. Without this,
        # async runs would have the frontend on the analytic pose path while
        # the backend still pays three torch.linalg.inv calls per render - and
        # mapping renders a great deal more than tracking does.
        #
        # This is a no-op in inline mode (same process, flag already set) and
        # matters only with use_inline_backend: False.
        _train = self.config.get("Training", {})
        if _train.get("iteration_graph", {}).get("enabled", False) or _train.get(
            "capture_safe_pose", False
        ):
            set_capture_safe_pose(True)

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.single_thread = bool(self.config.get("Dataset", {}).get(
            "single_thread", False))
        self.use_inline_backend = bool(self.config.get("Training", {}).get(
            "use_inline_backend", False))
        # INLINE is consumed by slam.py before BackEnd is constructed.  Read it
        # here as well: the environment override changes slam.py's attribute,
        # not the shared config dictionary, so otherwise the backend would
        # incorrectly retain the YAML value and select the async ceiling.
        if "INLINE" in os.environ:
            self.use_inline_backend = os.environ["INLINE"] not in (
                "0", "", "false", "False")

        self.mapping_iteration_ceiling = (
            self.mapping_itr_num
            if (self.single_thread or self.use_inline_backend)
            else min(10, self.mapping_itr_num)
        )
        _am_cfg, _am_bounds = autoscale_iteration_bounds(
            self.config.get("Training", {}).get("adaptive_mapping", {}),
            self.mapping_iteration_ceiling,
        )
        if _am_bounds["enabled"]:
            _mode = (
                "inline" if self.use_inline_backend
                else "single-thread" if self.single_thread
                else "async"
            )
            _old = _am_bounds["configured_bounds"]
            _new = _am_bounds["selected_bounds"]
            print(
                f"[AdaptiveMapper] MonoGS startup bounds: mode={_mode}, "
                f"mapping ceiling={self.mapping_iteration_ceiling}, "
                f"configured={_old[0]}..{_old[1]} at ceiling "
                f"{_am_bounds['reference_ceiling']}, "
                f"selected={_new[0]}..{_new[1]}" +
                (" (auto-scaled)" if _am_bounds["scaled"] else ""),
                flush=True,
            )
        self.adaptive_mapper = AdaptiveMapper(_am_cfg)
        # Novelty ratio (budget/ceiling) from the most recent keyframe-triggered
        # adaptive-mapping computation, reused to throttle the continuous
        # background mapping loop (see run()) between keyframe events - not
        # recomputed there, since that loop runs continuously and a probe on
        # every pass would cost as much as the mapping it's meant to save.
        # 1.0 = no throttle (default, and always the case when disabled).
        self._continuous_map_ratio = 1.0
        self._continuous_map_credit = 0.0
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        # Adaptive pruning (ported from RTGS, github.com/UMN-ZhaoLab/RTGS,
        # MonoGS_fullend branch) - gradient-informed, protected, late-
        # starting, rate-capped Gaussian removal after each keyframe's
        # mapping pass. See gaussian_model.py:adaptive_pruning().
        _ap_cfg = self.config.get("Training", {}).get("adaptive_pruning", {})
        self.adaptive_pruning_enabled = _ap_cfg.get("enabled", False)
        self.adaptive_pruning_ratio = _ap_cfg.get("target_reduction_ratio", 0.10)
        self.adaptive_pruning_start_progress = _ap_cfg.get("start_progress", 0.75)
        self.adaptive_pruning_max_step_frac = _ap_cfg.get(
            "max_prune_fraction_per_step", 0.04
        )
        self.adaptive_pruning_min_obs = _ap_cfg.get("min_mapping_observations", 2)
        self.adaptive_pruning_grad_keep_pct = _ap_cfg.get(
            "gradient_keep_percentile", 0.6
        )
        self.adaptive_pruning_min_opacity_protect = _ap_cfg.get(
            "min_opacity_protect", 0.85
        )
        self.adaptive_pruning_keyframe_interval = _ap_cfg.get("keyframe_interval", 1)
        self.adaptive_pruning_dataset_frames = _ap_cfg.get("dataset_frames", 592)
        self.adaptive_pruning_ramp_floor = _ap_cfg.get("ramp_floor", 0.25)
        self.adaptive_pruning_n_obs_protect = _ap_cfg.get("n_obs_protect", 3.0)
        self._adaptive_prune_keyframe_count = 0
        self.idle_mapping_prune_interval = self.config["Training"].get(
            "idle_mapping_prune_interval", 10
        )

    def post_mapping_adaptive_prune(self, cur_frame_idx, protected_kf_ids=None):
        if not self.adaptive_pruning_enabled or not self.gaussians:
            return 0
        self._adaptive_prune_keyframe_count += 1
        if (
            self._adaptive_prune_keyframe_count
            % self.adaptive_pruning_keyframe_interval
            != 0
        ):
            return 0
        before = self.gaussians.get_xyz.shape[0]
        try:
            removed, prune_mask = self.gaussians.adaptive_pruning(
                target_reduction_ratio=self.adaptive_pruning_ratio,
                frame_idx=cur_frame_idx,
                total_frames=self.adaptive_pruning_dataset_frames,
                pruning_start_progress=self.adaptive_pruning_start_progress,
                max_prune_fraction_per_step=self.adaptive_pruning_max_step_frac,
                min_mapping_observations=self.adaptive_pruning_min_obs,
                gradient_keep_percentile=self.adaptive_pruning_grad_keep_pct,
                min_opacity_protect=self.adaptive_pruning_min_opacity_protect,
                protected_kf_ids=protected_kf_ids,
                ramp_floor=self.adaptive_pruning_ramp_floor,
                n_obs_protect=self.adaptive_pruning_n_obs_protect,
            )
            if prune_mask is not None:
                # prune_points() only reindexes tensors owned by
                # GaussianModel - occ_aware_visibility is backend-side,
                # per-Gaussian, and was just built (pre-prune-sized) a few
                # lines above in map(). Reslice it the same way the
                # existing covisibility-based prune_mode='slam' branch
                # already does for its own to_prune mask, or the next
                # is_keyframe() call crashes on a tensor-size mismatch.
                keep_mask = ~prune_mask
                for kf_idx in list(self.occ_aware_visibility.keys()):
                    vis = self.occ_aware_visibility[kf_idx]
                    if vis.shape[0] == keep_mask.shape[0]:
                        self.occ_aware_visibility[kf_idx] = vis[keep_mask.to(vis.device)]
            after = self.gaussians.get_xyz.shape[0]
            if removed:
                Log(
                    f"Adaptive pruning after mapping frame {cur_frame_idx}: "
                    f"{before} -> {after} gaussians (removed={removed}, "
                    f"ref={getattr(self.gaussians, 'reference_gaussian_count', after)}, "
                    f"ratio={self.adaptive_pruning_ratio})"
                )
            return removed or 0
        except Exception as e:
            Log(f"Adaptive pruning failed: {e}")
            return 0

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1, return_first_loss=False,
            max_keyframes=None, max_random_viewpoints=None):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        # Render-workload cap (from AdaptiveMapper.compute_render_caps(),
        # optional/None otherwise): render only the freshest max_keyframes
        # of the window and up to max_random_viewpoints random viewpoints,
        # instead of always the full window + 2. The occ_aware_visibility
        # bookkeeping below is scoped to whichever subset was actually
        # rendered (current_window_render), not the full window, so it
        # can't index past what n_touched_acm actually contains. Pruning
        # eligibility (prune=True) is never called with a cap - it always
        # gets the full, uncapped window, so correctness there is
        # unaffected regardless of what capped iterations did.
        current_window_render = (
            current_window[:max_keyframes] if max_keyframes is not None
            else current_window
        )
        n_random = max_random_viewpoints if max_random_viewpoints is not None else 2

        _probe_depth_err = 0.0
        _probe_color_err = 0.0

        _map_t0 = time.time()
        for _iter_i in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window_render)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

                # Capture iter-0 depth/color error of current keyframe (first in window)
                if _iter_i == 0 and return_first_loss and cam_idx == 0:
                    if viewpoint.depth is not None:
                        # viewpoint.depth is a raw numpy array (unlike original_image,
                        # which is already a tensor - see get_loss_tracking_rgbd for
                        # the same conversion pattern used elsewhere in this file).
                        _gt_depth = torch.from_numpy(viewpoint.depth).to(
                            dtype=torch.float32, device=depth.device
                        )[None]
                        _valid = _gt_depth > 0
                        if _valid.any():
                            _probe_depth_err = (depth.detach()[_valid] - _gt_depth[_valid]).abs().mean().item()
                    _probe_color_err = (image.detach() - viewpoint.original_image.cuda()).abs().mean().item()

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:n_random]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range(len(current_window_render)):
                    kf_idx = current_window_render[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        self._mapping_time_total += time.time() - _map_t0
        self._mapping_iters_total += iters
        self._mlsys_map_calls += 1
        if return_first_loss:
            return gaussian_split, _probe_depth_err, _probe_color_err
        return gaussian_split

    def mapping_summary(self) -> str:
        """Aggregate mapping cost, in the same shape as Gaussian-SLAM's
        Mapper.mapping_summary() - one extractor regex covers both models.
        """
        if self._mapping_iters_total == 0 or self._mapping_frames_total == 0:
            return "Mapping time: no frames mapped"
        return (
            f"Mapping time: {self._mapping_time_total:.1f}s over "
            f"{self._mapping_iters_total} iterations across "
            f"{self._mapping_frames_total} frames "
            f"({1000 * self._mapping_time_total / self._mapping_iters_total:.2f} "
            f"ms/iter, "
            f"{self._mapping_iters_total / self._mapping_frames_total:.1f} iters/frame)"
        )

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def idle_mapping(self):
        """
        Substitute for run()'s continuous background-mapping loop when
        use_inline_backend is on - that loop lives inside run()'s own
        while True and never executes in inline mode (nothing drives it).
        Called directly by FrontEnd.run() (see slam_frontend.py) between
        keyframe events. Ported from RTGS (github.com/UMN-ZhaoLab/RTGS,
        MonoGS_fullend branch, slam_backend.py:idle_mapping()).
        """
        if self.pause or len(self.current_window) == 0:
            return
        if self.use_inline_backend:
            self.map(self.current_window)
            if self.last_sent >= self.idle_mapping_prune_interval:
                self.map(self.current_window, prune=True, iters=10)
                self.push_to_frontend()

    def process_message(self, data):
        """
        Handle one backend_queue message. Extracted out of run() so it can
        also be called directly, synchronously, from InlineBackendQueue.put()
        for the use_inline_backend concurrency mode (no separate process/
        thread) - see multiprocessing_utils.py. Returns True on "stop" so
        the caller knows to break its loop; run()'s own continuous-mapping
        branch (the queue-empty case) is untouched and lives only in run().
        """
        if data[0] == "stop":
            return True
        elif data[0] == "pause":
            self.pause = True
        elif data[0] == "unpause":
            self.pause = False
        elif data[0] == "color_refinement":
            # Print here, not only at "stop": color_refinement is a
            # ~3.5min/26000-iteration fixed post-process that runs
            # after all real mapping work is done, so waiting for
            # "stop" (which arrives even later) just delays seeing
            # numbers that are already final by this point.
            Log(self.adaptive_mapper.summary())
            self.color_refinement()
            self.push_to_frontend()
        elif data[0] == "init":
            cur_frame_idx = data[1]
            viewpoint = data[2]
            depth_map = data[3]
            self._mapping_frames_total += 1
            Log("Resetting the system")
            self.reset()

            self.viewpoints[cur_frame_idx] = viewpoint
            self.add_next_kf(
                cur_frame_idx, viewpoint, depth_map=depth_map, init=True
            )
            self.initialize_map(cur_frame_idx, viewpoint)
            self.push_to_frontend("init")

        elif data[0] == "keyframe":
            cur_frame_idx = data[1]
            viewpoint = data[2]
            current_window = data[3]
            depth_map = data[4]
            self._mapping_frames_total += 1

            self.viewpoints[cur_frame_idx] = viewpoint
            self.current_window = current_window
            _count_before_kf = self.gaussians.get_xyz.shape[0]
            self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
            _added_gaussians = self.gaussians.get_xyz.shape[0] - _count_before_kf

            opt_params = []
            frames_to_optimize = self.config["Training"]["pose_window"]
            # Full mapping_itr_num (not just the async ceiling of 10) when
            # single_thread OR use_inline_backend - both eliminate the
            # continuous background-mapping loop (it lives in run(), which
            # never executes in either mode), so each keyframe's own pass
            # needs to be deep enough to compensate on its own. Confirmed
            # this is the actual mechanism RTGS's single_thread ("RTGS
            # Soft") recipe relies on - their use_inline_backend path kept
            # the shallow per-keyframe budget instead and needed
            # idle_mapping() to compensate, which their own published
            # numbers show doesn't fully make up the gap (see
            # exp_adaptive_pruning / rtgs_concurrency_and_benchmark_caveat
            # memory).
            # This is also the ceiling used to auto-scale AdaptiveMapper's
            # configured min/max bounds during startup.  Keeping one selected
            # value prevents the logged controller range from drifting away
            # from the budget that is actually executed.
            iter_per_kf = self.mapping_iteration_ceiling
            if not self.initialized:
                if (
                    len(self.current_window)
                    == self.config["Training"]["window_size"]
                ):
                    frames_to_optimize = (
                        self.config["Training"]["window_size"] - 1
                    )
                    iter_per_kf = 50 if self.live_mode else 300
                    Log("Performing initial BA for initialization")
                else:
                    iter_per_kf = self.mapping_itr_num
            for cam_idx in range(len(self.current_window)):
                if self.current_window[cam_idx] == 0:
                    continue
                viewpoint = self.viewpoints[current_window[cam_idx]]
                if cam_idx < frames_to_optimize:
                    opt_params.append(
                        {
                            "params": [viewpoint.cam_rot_delta],
                            "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                            * 0.5,
                            "name": "rot_{}".format(viewpoint.uid),
                        }
                    )
                    opt_params.append(
                        {
                            "params": [viewpoint.cam_trans_delta],
                            "lr": self.config["Training"]["lr"][
                                "cam_trans_delta"
                            ]
                            * 0.5,
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
            self.keyframe_optimizers = torch.optim.Adam(opt_params)

            if self.adaptive_mapper.enabled and self.initialized:
                # Gate on self.initialized only, not self.single_thread:
                # iter_per_kf is already single_thread-aware (full
                # mapping_itr_num when sync, the async per-keyframe
                # ceiling of 10 otherwise), so clamping the computed
                # budget to iter_per_kf makes this work correctly in
                # either threading mode - only the *ceiling* changes.
                # AdaptiveMapper's min/max bounds were scaled to this ceiling
                # in set_hyperparams(), before the first mapping event.
                _, _d_err, _c_err = self.map(
                    self.current_window, iters=1, return_first_loss=True)
                _budget = min(
                    self.adaptive_mapper.compute_budget(
                        depth_error=_d_err, color_error=_c_err),
                    iter_per_kf,
                )
                # Cache for the continuous background loop (see run()) -
                # that loop doesn't get its own keyframe event to
                # trigger a fresh probe, so it rides on this reading
                # until the next keyframe updates it.
                self._continuous_map_ratio = _budget / iter_per_kf
                if _budget > 1:
                    # Render-workload cap for the budgeted iterations
                    # only - the probe call above stays uncapped (an
                    # accurate depth/color novelty reading needs the
                    # full window) and the final prune=True call
                    # below is never capped either (pruning
                    # eligibility needs the full window).
                    _max_kf, _max_rv = self.adaptive_mapper.compute_render_caps(
                        len(self.current_window))
                    self.map(
                        self.current_window, iters=_budget - 1,
                        max_keyframes=_max_kf, max_random_viewpoints=_max_rv,
                    )
            else:
                self.map(self.current_window, iters=iter_per_kf)
            self.map(self.current_window, prune=True)
            self.gaussians.register_keyframe_growth(_added_gaussians)
            _protected_kf_ids = list(self.current_window)
            if cur_frame_idx not in _protected_kf_ids:
                _protected_kf_ids.append(cur_frame_idx)
            self.post_mapping_adaptive_prune(
                cur_frame_idx, protected_kf_ids=_protected_kf_ids
            )
            self.push_to_frontend("keyframe")
        else:
            raise Exception("Unprocessed data", data)
        return False

    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                # Throttle background mapping by the novelty ratio cached from
                # the most recent keyframe event (see the "keyframe" branch
                # in process_message()) - a deterministic credit accumulator,
                # not a probe on every pass: at ratio=1.0 (disabled, or
                # high-novelty window) this runs every single loop pass
                # exactly like before; at a lower ratio it runs
                # proportionally less often. No sleep on skip - tried that
                # first, made things ~15% slower overall, because each
                # skipped map() call was cheap (iters=1, small window) but
                # time.sleep()'s actual OS-level granularity on this VM is
                # apparently much coarser than the requested 1ms, so the
                # sleep overhead exceeded the GPU work it saved. Just loop
                # back and recheck the queue immediately instead - CPU
                # cycles here are cheap, that was never what we're saving.
                self._continuous_map_credit += self._continuous_map_ratio
                if self._continuous_map_credit >= 1.0:
                    self._continuous_map_credit -= 1.0
                    self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if self.process_message(data):
                    break
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        Log(self.adaptive_mapper.summary())
        Log(self.mapping_summary())
        # mlsys CSV: mapping cost per map() call (single_thread only -- in multi-thread the
        # backend is a separate process and the other MonoGS repos leave these columns blank).
        if self.single_thread and self._mlsys_map_calls > 0:
            Log(f"Mapping ms/frame: {1000.0 * self._mapping_time_total / self._mlsys_map_calls:.4f}", tag="Eval")
            Log(f"Mapping ms/iter: "
                f"{1000.0 * self._mapping_time_total / max(self._mapping_iters_total, 1):.4f}", tag="Eval")
            Log(f"Mapping iters/frame: {self._mapping_iters_total / self._mlsys_map_calls:.4f}", tag="Eval")
        return
