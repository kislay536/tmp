""" This module includes the Mapper class, which is responsible scene mapping: Paragraph 3.2  """
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision

from src.entities.arguments import OptimizationParams
from src.entities.datasets import TUM_RGBD, BaseDataset, ScanNet
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.losses import isotropic_loss, l1_loss, ssim
from src.utils.mapper_utils import (calc_psnr, compute_camera_frustum_corners,
                                    compute_frustum_point_ids,
                                    compute_new_points_ids,
                                    compute_opt_views_distribution,
                                    create_point_cloud, geometric_edge_mask,
                                    sample_pixels_based_on_gradient)
from src.utils.utils import (get_render_settings, np2ptcloud, np2torch,
                             render_gaussian_model, torch2np)
from src.utils.vis_utils import *  # noqa - needed for debugging

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..')))
from utils.adaptive_mapper import AdaptiveMapper
from utils.map_iter_trace import MapTrace


class Mapper(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:
        """ Sets up the mapper parameters
        Args:
            config: configuration of the mapper
            dataset: The dataset object used for extracting camera parameters and reading the data
            logger: The logger object used for logging the mapping process and saving visualizations
        """
        self.config = config
        self.logger = logger
        self.dataset = dataset
        self.iterations = config["iterations"]
        # MAP_ITERS overrides the regular-frame mapping cap without editing
        # the YAML, same convention as tracker.py's TRACK_ITERS - for the
        # budget-vs-quality ablation (profiling/legacy/run_gslam_mapping_budget_curve.sh).
        # new_submap_iterations is deliberately untouched: those are the
        # once-per-submap seeding frames, a different question from "does a
        # REGULAR frame need this many mapping iterations".
        if "MAP_ITERS" in os.environ:
            self.iterations = int(os.environ["MAP_ITERS"])
        # LOUD AND UNCONDITIONAL, not just on override: the budget sweep's
        # own "100.0 iters/frame" summary line looked identical across
        # different MAP_ITERS values on wks-26-0003, and that summary is an
        # ACCUMULATED runtime counter (mapping_summary() below), not a
        # static config read - so either the override isn't reaching this
        # constructor, or something downstream of self.iterations isn't
        # using it. This print settles which, directly in the log, instead
        # of guessing further from code reading alone.
        print(f"[Mapper] regular-frame self.iterations={self.iterations} "
              f"(config default was {config['iterations']}, "
              f"MAP_ITERS in environ: {'MAP_ITERS' in os.environ}, "
              f"raw value: {os.environ.get('MAP_ITERS')!r})", flush=True)
        self.new_submap_iterations = config["new_submap_iterations"]
        self.new_submap_points_num = config["new_submap_points_num"]
        self.new_submap_gradient_points_num = config["new_submap_gradient_points_num"]
        self.new_frame_sample_size = config["new_frame_sample_size"]
        self.new_points_radius = config["new_points_radius"]
        self.alpha_thre = config["alpha_thre"]
        self.pruning_thre = config["pruning_thre"]
        self.current_view_opt_iterations = config["current_view_opt_iterations"]
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.keyframes = []
        self.map_trace = MapTrace(config.get("iter_trace", {}))
        _am = dict(config.get("adaptive_mapping", {}))
        # ADMAP=0/1 without editing the YAML, same name the Replica config uses.
        # Mapping DOMINATES wall time on Gaussian-SLAM - tracking was 280s of
        # 1521s in the first preconditioner run - so a tracking speedup is
        # heavily diluted unless the mapping side is attacked too. The recorded
        # win here is 1.92x from adaptive mapping alone
        # (results/adaptive_mapping_results_clean.txt).
        if "ADMAP" in os.environ:
            _am["enabled"] = os.environ["ADMAP"] not in ("0", "", "false", "False")
        # Per-run mapping-floor override.  Keep this separate from Mapper's
        # native iteration cap: it changes only AdaptiveMapper's lower bound,
        # while new-submap frames still use new_submap_iterations in map().
        if "AM_MIN_ITERS" in os.environ:
            _am_min_iters = int(os.environ["AM_MIN_ITERS"])
            _am_max_iters = int(_am.get("max_iters", self.iterations))
            if not 0 <= _am_min_iters <= _am_max_iters:
                raise ValueError(
                    f"AM_MIN_ITERS={_am_min_iters} must be between 0 and "
                    f"adaptive_mapping.max_iters={_am_max_iters}"
                )
            _am["min_iters"] = _am_min_iters
        # Per-run novelty-weight overrides.  GSLAM supplies only unseen
        # coverage and newly seeded points, so these two weights control the
        # complete adaptive-mapping mixture without changing accepted YAMLs.
        _am_weight_overridden = False
        for _env_name, _cfg_name in (
            ("AM_W_UNSEEN", "w_unseen"),
            ("AM_W_NEW_PTS", "w_new_pts"),
        ):
            if _env_name in os.environ:
                _weight = float(os.environ[_env_name])
                if not np.isfinite(_weight) or _weight < 0.0:
                    raise ValueError(
                        f"{_env_name}={os.environ[_env_name]!r} must be a "
                        "finite, non-negative number"
                    )
                _am[_cfg_name] = _weight
                _am_weight_overridden = True
        if _am_weight_overridden:
            _w_unseen = float(_am.get("w_unseen", 0.0))
            _w_new_pts = float(_am.get("w_new_pts", 0.0))
            if _w_unseen + _w_new_pts <= 0.0:
                raise ValueError(
                    "AM_W_UNSEEN and AM_W_NEW_PTS cannot both be zero"
                )
            print(
                "[Mapper] adaptive mapping weights overridden: "
                f"unseen={_w_unseen:g}, new_pts={_w_new_pts:g}",
                flush=True,
            )
        # GSLAM adapts from unseen coverage and the raw number of newly added
        # Gaussians.  Expose the matching calibration window so a run can fit
        # n_new_pts_ref on its own scene instead of carrying TUM's fixed 5000.
        # Depth/colour calibration is intentionally not enabled here: those
        # signals are absent when their configured weights are zero.
        if "AM_CALIBRATE_NEW_PTS_FRAMES" in os.environ:
            _am_calib_frames = int(os.environ["AM_CALIBRATE_NEW_PTS_FRAMES"])
            if _am_calib_frames < 0:
                raise ValueError("AM_CALIBRATE_NEW_PTS_FRAMES must be non-negative")
            _am["calibrate_new_pts_frames"] = _am_calib_frames
        if "AM_CALIBRATE_NEW_PTS_SKIP" in os.environ:
            _am_calib_skip = int(os.environ["AM_CALIBRATE_NEW_PTS_SKIP"])
            if _am_calib_skip < 0:
                raise ValueError("AM_CALIBRATE_NEW_PTS_SKIP must be non-negative")
            _am["calibrate_new_pts_skip"] = _am_calib_skip
        if "AM_CALIBRATE_NEW_PTS_SCALE" in os.environ:
            _am_calib_scale = float(os.environ["AM_CALIBRATE_NEW_PTS_SCALE"])
            if not np.isfinite(_am_calib_scale) or _am_calib_scale <= 0.0:
                raise ValueError(
                    "AM_CALIBRATE_NEW_PTS_SCALE must be a finite number "
                    "greater than zero"
                )
            _am["calibrate_new_pts_scale"] = _am_calib_scale
        # AM_NEWPTS_REF overrides the novelty denominator without editing a
        # config, because it is the one value here that does not transfer.
        #
        # n_new_pts_ratio = min(new_pts / n_new_pts_ref, 1.0), so a reference
        # BELOW the scene's typical new-point count saturates the ratio at 1.0
        # on every frame: novelty pins high, the budget never leaves max_iters,
        # and adaptive mapping is inert WHILE LOGGING AS ENABLED. TUM ships
        # 5000; a Replica room0 frame logged 44011 points added, so the shipped
        # value is ~9x too small there and the Replica suite configs carry
        # 40000 instead.
        #
        # AdaptiveMapper's own default is 300, which is what an absent
        # adaptive_mapping block would supply - another reason the Replica
        # chain needed the block restated rather than the flag flipped.
        #
        # Read the per-frame "[AdaptiveMap] ... new_pts=X" column: pinned at
        # 1.000 means still too small, near 0 means too large.
        if "AM_NEWPTS_REF" in os.environ:
            _am["n_new_pts_ref"] = int(os.environ["AM_NEWPTS_REF"])
        self.adaptive_mapper = AdaptiveMapper(_am)
        # SKIP_VIS removes an extra render, PSNR evaluation, and matplotlib
        # write.  Mapping-budget text is useful independently of that costly
        # diagnostic path, so let runs restore only the lightweight log.
        self.mapping_log = os.environ.get("GSLAM_MAPPING_LOG", "0") not in (
            "0", "", "false", "False"
        )
        if self.mapping_log:
            print("[Mapper] lightweight mapping log enabled", flush=True)

        # MAPPING-SIDE TIME/ITERATION TOTALS, symmetric to the tracker's
        # _tracking_time_total (tracker.py:511) - added for experiments/'s
        # mapping_ms_per_frame / mapping_ms_per_iter / mapping_iters_per_frame
        # columns, which had no aggregate source anywhere in this file before.
        # optimization_time is already measured per call (line below); this
        # only accumulates it, it does not change what map() does or returns.
        self._mapping_time_total = 0.0
        self._mapping_iters_total = 0
        self._mapping_frames = 0

    def mapping_summary(self) -> str:
        """Aggregate mapping cost, in the same shape as tracker.tracking_summary()."""
        if self._mapping_iters_total == 0 or self._mapping_frames == 0:
            return "Mapping time: no frames mapped"
        return (
            f"Mapping time: {self._mapping_time_total:.1f}s over "
            f"{self._mapping_iters_total} iterations across "
            f"{self._mapping_frames} frames "
            f"({1000 * self._mapping_time_total / self._mapping_iters_total:.2f} "
            f"ms/iter, "
            f"{self._mapping_iters_total / self._mapping_frames:.1f} iters/frame)"
        )

    def compute_seeding_mask(self, gaussian_model: GaussianModel, keyframe: dict, new_submap: bool) -> np.ndarray:
        """
        Computes a binary mask to identify regions within a keyframe where new Gaussian models should be seeded
        based on alpha masks or color gradient
        Args:
            gaussian_model: The current submap
            keyframe (dict): Keyframe dict containing color, depth, and render settings
            new_submap (bool): A boolean indicating whether the seeding is occurring in current submap or a new submap
        Returns:
            np.ndarray: A binary mask of shpae (H, W) indicates regions suitable for seeding new 3D Gaussian models
        """
        seeding_mask = None
        if new_submap:
            color_for_mask = (torch2np(keyframe["color"].permute(1, 2, 0)) * 255).astype(np.uint8)
            seeding_mask = geometric_edge_mask(color_for_mask, RGB=True)
        else:
            render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            alpha_mask = (render_dict["alpha"] < self.alpha_thre)
            gt_depth_tensor = keyframe["depth"][None]
            depth_error = torch.abs(gt_depth_tensor - render_dict["depth"]) * (gt_depth_tensor > 0)
            depth_error_mask = (render_dict["depth"] > gt_depth_tensor) * (depth_error > 40 * depth_error.median())
            seeding_mask = alpha_mask | depth_error_mask
            seeding_mask = torch2np(seeding_mask[0])
        return seeding_mask

    def seed_new_gaussians(self, gt_color: np.ndarray, gt_depth: np.ndarray, intrinsics: np.ndarray,
                           estimate_c2w: np.ndarray, seeding_mask: np.ndarray, is_new_submap: bool) -> np.ndarray:
        """
        Seeds means for the new 3D Gaussian based on ground truth color and depth, camera intrinsics,
        estimated camera-to-world transformation, a seeding mask, and a flag indicating whether this is a new submap.
        Args:
            gt_color: The ground truth color image as a numpy array with shape (H, W, 3).
            gt_depth: The ground truth depth map as a numpy array with shape (H, W).
            intrinsics: The camera intrinsics matrix as a numpy array with shape (3, 3).
            estimate_c2w: The estimated camera-to-world transformation matrix as a numpy array with shape (4, 4).
            seeding_mask: A binary mask indicating where to seed new Gaussians, with shape (H, W).
            is_new_submap: Flag indicating whether the seeding is for a new submap (True) or an existing submap (False).
        Returns:
            np.ndarray: An array of 3D points where new Gaussians will be initialized, with shape (N, 3)

        """
        pts = create_point_cloud(gt_color, 1.005 * gt_depth, intrinsics, estimate_c2w)
        flat_gt_depth = gt_depth.flatten()
        non_zero_depth_mask = flat_gt_depth > 0.  # need filter if zero depth pixels in gt_depth
        valid_ids = np.flatnonzero(seeding_mask)
        if is_new_submap:
            if self.new_submap_points_num < 0:
                uniform_ids = np.arange(pts.shape[0])
            else:
                uniform_ids = np.random.choice(pts.shape[0], self.new_submap_points_num, replace=False)
            gradient_ids = sample_pixels_based_on_gradient(gt_color, self.new_submap_gradient_points_num)
            combined_ids = np.concatenate((uniform_ids, gradient_ids))
            combined_ids = np.concatenate((combined_ids, valid_ids))
            sample_ids = np.unique(combined_ids)
        else:
            if self.new_frame_sample_size < 0 or len(valid_ids) < self.new_frame_sample_size:
                sample_ids = valid_ids
            else:
                sample_ids = np.random.choice(valid_ids, size=self.new_frame_sample_size, replace=False)
        sample_ids = sample_ids[non_zero_depth_mask[sample_ids]]
        return pts[sample_ids, :].astype(np.float32)

    def optimize_submap(self, keyframes: list, gaussian_model: GaussianModel, iterations: int = 100,
                        n_new_pts=None, unseen_ratio=None) -> dict:
        """
        Optimizes the submap by refining the parameters of the 3D Gaussian based on the observations
        from keyframes observing the submap.
        Args:
            keyframes: A list of tuples consisting of frame id and keyframe dictionary
            gaussian_model: An instance of the GaussianModel class representing the initial state
                of the Gaussian model to be optimized.
            iterations: The number of iterations to perform the optimization process. Defaults to 100.
        Returns:
            losses_dict: Dictionary with the optimization statistics
        """

        iteration = 0
        losses_dict = {}

        current_frame_iters = self.current_view_opt_iterations * iterations
        distribution = compute_opt_views_distribution(len(keyframes), iterations, current_frame_iters)
        start_time = time.time()
        # BUG (found 2026-09-30): this was keyframes[-1][0]. The caller
        # always passes keyframes = [(frame_id, keyframe)] + self.keyframes
        # (map()'s call site), so the CURRENT frame is keyframes[0], not
        # keyframes[-1] - that's the most recently APPENDED entry in
        # self.keyframes, i.e. the PREVIOUS frame's keyframe for any regular
        # round. Every regular round's trace row was mislabeled with the
        # prior frame's id; it only happened to be correct for a
        # new-submap round, where self.keyframes was just cleared so
        # keyframes has exactly one entry. Confirmed empirically: a 200-row
        # trace had only 194 unique "frame" values - 6 collisions between a
        # correctly-labeled new-submap round and a mislabeled regular round
        # sharing its id, silently dropping the new-submap round whenever
        # the two got deduplicated by frame id downstream.
        self.map_trace.begin_frame(keyframes[0][0], iterations,
                                   n_new_pts=n_new_pts, unseen_ratio=unseen_ratio)
        while iteration < iterations + 1:
            gaussian_model.optimizer.zero_grad(set_to_none=True)
            keyframe_id = np.random.choice(np.arange(len(keyframes)), p=distribution)

            frame_id, keyframe = keyframes[keyframe_id]
            render_pkg = render_gaussian_model(gaussian_model, keyframe["render_settings"])

            image, depth = render_pkg["color"], render_pkg["depth"]
            gt_image = keyframe["color"]
            gt_depth = keyframe["depth"]

            mask = (gt_depth > 0) & (~torch.isnan(depth)).squeeze(0)
            color_loss = (1.0 - self.opt.lambda_dssim) * l1_loss(
                image[:, mask], gt_image[:, mask]) + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            depth_loss = l1_loss(depth[:, mask], gt_depth[mask])
            reg_loss = isotropic_loss(gaussian_model.get_scaling())
            total_loss = color_loss + depth_loss + reg_loss
            if self.map_trace.active:
                self.map_trace.capture(total_loss.detach(), frame_id)
            total_loss.backward()

            losses_dict[frame_id] = {"color_loss": color_loss.item(),
                                     "depth_loss": depth_loss.item(),
                                     "total_loss": total_loss.item()}

            with torch.no_grad():

                if iteration == iterations // 2 or iteration == iterations:
                    prune_mask = (gaussian_model.get_opacity()
                                  < self.pruning_thre).squeeze()
                    gaussian_model.prune_points(prune_mask)

                # Optimizer step
                if iteration < iterations:
                    gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none=True)

            iteration += 1
        self.map_trace.end_frame()
        optimization_time = time.time() - start_time
        losses_dict["optimization_time"] = optimization_time
        losses_dict["optimization_iter_time"] = optimization_time / iterations
        return losses_dict

    def grow_submap(self, gt_depth: np.ndarray, estimate_c2w: np.ndarray, gaussian_model: GaussianModel,
                    pts: np.ndarray, filter_cloud: bool) -> int:
        """
        Expands the submap by integrating new points from the current keyframe
        Args:
            gt_depth: The ground truth depth map for the current keyframe, as a 2D numpy array.
            estimate_c2w: The estimated camera-to-world transformation matrix for the current keyframe of shape (4x4)
            gaussian_model (GaussianModel): The Gaussian model representing the current state of the submap.
            pts: The current set of 3D points in the keyframe of shape (N, 3)
            filter_cloud: A boolean flag indicating whether to apply filtering to the point cloud to remove
                outliers or noise before integrating it into the map.
        Returns:
            int: The number of points added to the submap
        """
        gaussian_points = gaussian_model.get_xyz()
        camera_frustum_corners = compute_camera_frustum_corners(gt_depth, estimate_c2w, self.dataset.intrinsics)
        reused_pts_ids = compute_frustum_point_ids(
            gaussian_points, np2torch(camera_frustum_corners), device="cuda")
        new_pts_ids = compute_new_points_ids(gaussian_points[reused_pts_ids], np2torch(pts[:, :3]).contiguous(),
                                             radius=self.new_points_radius, device="cuda")
        new_pts_ids = torch2np(new_pts_ids)
        if new_pts_ids.shape[0] > 0:
            cloud_to_add = np2ptcloud(pts[new_pts_ids, :3], pts[new_pts_ids, 3:] / 255.0)
            if filter_cloud:
                cloud_to_add, _ = cloud_to_add.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
            gaussian_model.add_points(cloud_to_add)
        gaussian_model._features_dc.requires_grad = False
        gaussian_model._features_rest.requires_grad = False
        if (self.mapping_log
                or os.environ.get("GSLAM_SKIP_VIS", "0") != "1"):
            print("Gaussian model size", gaussian_model.get_size())
        return new_pts_ids.shape[0]

    def map(self, frame_id: int, estimate_c2w: np.ndarray, gaussian_model: GaussianModel, is_new_submap: bool) -> dict:
        """ Calls out the mapping process described in paragraph 3.2
        The process goes as follows: seed new gaussians -> add to the submap -> optimize the submap
        Args:
            frame_id: current keyframe id
            estimate_c2w (np.ndarray): The estimated camera-to-world transformation matrix of shape (4x4)
            gaussian_model (GaussianModel): The current Gaussian model of the submap
            is_new_submap (bool): A boolean flag indicating whether the current frame initiates a new submap
        Returns:
            opt_dict: Dictionary with statistics about the optimization process
        """

        _, gt_color, gt_depth, _ = self.dataset[frame_id]
        estimate_w2c = np.linalg.inv(estimate_c2w)

        color_transform = torchvision.transforms.ToTensor()
        keyframe = {
            "color": color_transform(gt_color).cuda(),
            "depth": np2torch(gt_depth, device="cuda"),
            "render_settings": get_render_settings(
                self.dataset.width, self.dataset.height, self.dataset.intrinsics, estimate_w2c)}

        seeding_mask = self.compute_seeding_mask(gaussian_model, keyframe, is_new_submap)
        pts = self.seed_new_gaussians(
            gt_color, gt_depth, self.dataset.intrinsics, estimate_c2w, seeding_mask, is_new_submap)

        filter_cloud = isinstance(self.dataset, (TUM_RGBD, ScanNet)) and not is_new_submap

        new_pts_num = self.grow_submap(gt_depth, estimate_c2w, gaussian_model, pts, filter_cloud)

        max_iterations = self.iterations
        if is_new_submap:
            max_iterations = self.new_submap_iterations

        # Adaptive mapping: use seeding_mask coverage + new pts as novelty signals.
        # New-submap frames always run full new_submap_iterations - skip adaptation there.
        _unseen = None
        if not is_new_submap and seeding_mask is not None:
            _H, _W = gt_depth.shape[:2]
            _unseen = float(seeding_mask.sum()) / float(_H * _W)
            _n_new_ratio = min(float(new_pts_num) / max(self.adaptive_mapper.n_new_pts_ref, 1), 1.0)

            # Pre-optimization depth/color probe on the current keyframe, so
            # AdaptiveMapper can use w_depth/w_color too (previously this only
            # ever passed unseen_ratio/n_new_pts_ratio). Mirrors MonoGS's iter-0
            # probe (slam_backend.py, return_first_loss) and SplaTAM's iter-0
            # loss capture: read the error BEFORE any mapping step touches the
            # model, so the signal reflects novelty, not this frame's own
            # optimization progress. Plain L1, no SSIM/reg, to match those two.
            # Gated on the weights so this is a no-op (skips the extra render)
            # on every existing config, which ships w_depth=w_color=0.0.
            _depth_err = _color_err = None
            if self.adaptive_mapper.w_depth > 0.0 or self.adaptive_mapper.w_color > 0.0:
                with torch.no_grad():
                    probe_pkg = render_gaussian_model(gaussian_model, keyframe["render_settings"])
                    probe_depth = probe_pkg["depth"]
                    probe_mask = (keyframe["depth"] > 0) & (~torch.isnan(probe_depth)).squeeze(0)
                    if probe_mask.any():
                        _depth_err = l1_loss(probe_depth[:, probe_mask], keyframe["depth"][probe_mask]).item()
                    _color_err = l1_loss(probe_pkg["color"], keyframe["color"]).item()

            _budget = self.adaptive_mapper.compute_budget(
                unseen_ratio=_unseen,
                n_new_pts_ratio=_n_new_ratio,
                n_new_pts=new_pts_num,
                depth_error=_depth_err,
                color_error=_color_err,
            )
            if (self.adaptive_mapper.enabled and _budget < max_iterations
                    and (self.mapping_log
                         or os.environ.get("GSLAM_SKIP_VIS", "0") != "1")):
                print(f"  [AdaptiveMap] frame {frame_id}: {_budget}/{max_iterations} iters"
                      f" (unseen={_unseen:.3f} new_pts={_n_new_ratio:.3f})")
            # BUG (found 2026-09-29, via the mapping-budget sweep): this used
            # to be `map_iterations = _budget` unconditionally - but
            # AdaptiveMapper.compute_budget() returns ITS OWN self.max_iters
            # when disabled (utils/adaptive_mapper.py:124-125), which is a
            # SEPARATE config value from Mapper.iterations/max_iterations.
            # With ADMAP=0 every regular (non-new-submap) frame was silently
            # getting AdaptiveMapper's max_iters (config default, e.g. 100)
            # instead of self.iterations - MAP_ITERS had no effect on any of
            # them. Invisible before because nothing had ever made the two
            # configs diverge; MAP_ITERS was the first thing to.
            map_iterations = _budget if self.adaptive_mapper.enabled else max_iterations
        else:
            map_iterations = max_iterations

        start_time = time.time()
        opt_dict = self.optimize_submap([(frame_id, keyframe)] + self.keyframes, gaussian_model, map_iterations,
                                        n_new_pts=new_pts_num, unseen_ratio=_unseen)
        optimization_time = time.time() - start_time
        self._mapping_time_total += optimization_time
        self._mapping_iters_total += map_iterations
        self._mapping_frames += 1
        if (self.mapping_log
                or os.environ.get("GSLAM_SKIP_VIS", "0") != "1"):
            print("Optimization time: ", optimization_time)

        self.keyframes.append((frame_id, keyframe))

        # Visualise the mapping for the current frame.
        #
        # UNGATED AND EXPENSIVE, which is why GSLAM_SKIP_VIS exists. On EVERY
        # mapped frame this does an extra full render, a PSNR evaluation, and
        # then logger.vis_mapping_iteration writes a 250-DPI matplotlib figure
        # to disk. All of it is CPU- and disk-bound, so it leaves the GPU idle:
        # a profiled run measured 13-30% GPU utilisation and took over 40
        # minutes to reach launch 6000 under ncu, whose per-launch interception
        # compounds it.
        #
        # DEFAULT IS UNCHANGED. Skipping it would make this run incomparable
        # with every GSLAM number already recorded (2553.7s baseline, 1450.0s,
        # 1285.3s, 1238.4s), so the fast path is opt-in and must never be used
        # for a timing figure that is compared against those.
        #
        # psnr_render is still written when skipping, because
        # logger.log_mapping_iteration reads opt_dict["psnr_render"]
        # unconditionally and would raise KeyError. float('nan') marks it as
        # not measured rather than silently reporting 0.0 as a PSNR.
        if os.environ.get("GSLAM_SKIP_VIS", "0") == "1":
            opt_dict["psnr_render"] = float("nan")
        else:
            with torch.no_grad():
                render_pkg_vis = render_gaussian_model(gaussian_model, keyframe["render_settings"])
                image_vis, depth_vis = render_pkg_vis["color"], render_pkg_vis["depth"]
                psnr_value = calc_psnr(image_vis, keyframe["color"]).mean().item()
                opt_dict["psnr_render"] = psnr_value
                print(f"PSNR this frame: {psnr_value}")
                self.logger.vis_mapping_iteration(
                    frame_id, map_iterations,
                    image_vis.clone().detach().permute(1, 2, 0),
                    depth_vis.clone().detach().permute(1, 2, 0),
                    keyframe["color"].permute(1, 2, 0),
                    keyframe["depth"].unsqueeze(-1),
                    seeding_mask=seeding_mask)

        # Log the mapping numbers for the current frame
        self.logger.log_mapping_iteration(frame_id, new_pts_num, gaussian_model.get_size(),
                                          optimization_time/map_iterations, opt_dict)
        return opt_dict
