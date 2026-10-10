""" This module includes the Gaussian-SLAM class, which is responsible for controlling Mapper and Tracker
    It also decides when to start a new submap and when to update the estimated camera poses.
"""
import json
import os
import pprint
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.mapper import Mapper
from src.entities.tracker import Tracker

# Repo root on the path so `utils` resolves as the PEP 420 namespace package
# spanning repo-root/utils and this project's own. tracker.py does the same
# insert; repeating it here keeps this module importable on its own.
import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from utils.prefetch import FramePrefetcher  # noqa: E402
from src.entities.logger import Logger
from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.mapper_utils import exceeds_motion_thresholds
from src.utils.utils import np2torch, setup_seed, torch2np
from src.utils.vis_utils import *  # noqa - needed for debugging


class GaussianSLAM(object):

    def __init__(self, config: dict) -> None:

        self._setup_output_path(config)
        self.device = "cuda"
        self.config = config

        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        # frame_limit lives at the top level in TUM/Replica/ScanNet configs but inside
        # data: in ScanNetPP configs — merge both, preferring the data-level value.
        _dataset_cfg = {**config["data"], **config["cam"]}
        if "frame_limit" not in _dataset_cfg:
            _dataset_cfg["frame_limit"] = config.get("frame_limit", -1)
        # FRAMES=N truncates the sequence without editing a config, matching
        # SplaTAM's env name. Short runs are how an implementation bug gets
        # caught cheaply - the preconditioner port composed its step on top of
        # Adam's for six full-length arms before anyone ran a 250-frame
        # diagnostic, and every one of those numbers had to be thrown away.
        if "FRAMES" in os.environ:
            _dataset_cfg["frame_limit"] = int(os.environ["FRAMES"])
        self.dataset = get_dataset(config["dataset_name"])(_dataset_cfg)

        # Frame prefetch (utils/prefetch.py, shared with MonoGS and SplaTAM).
        # Default OFF.
        #
        # Gaussian-SLAM is the cleanest of the three to wrap: its __getitem__
        # returns numpy (index, color, depth, pose) and never touches CUDA, so
        # the worker-thread contract at the top of prefetch.py is satisfied
        # as-is - no to_device adapter, no dataset-on-CPU dance.
        #
        # `recent` is NOT optional here. The tracker reads dataset[frame_id]
        # and then dataset[frame_id - 1] (twice), and the mapper reads
        # dataset[frame_id] again. Against a forward-only queue every backward
        # read tears the worker down and rebuilds it, once per frame.
        _pf_cfg = config.get("prefetch", {})
        if _pf_cfg.get("enabled", False):
            _pf_cfg = {"recent": 2, **_pf_cfg}
            self.dataset = FramePrefetcher(self.dataset, _pf_cfg, name="GSLAM")
            print(f"[GaussianSLAM] frame prefetch ON "
                  f"(queue_depth={self.dataset.queue_depth}, "
                  f"recent={self.dataset.recent_size})")

        n_frames = len(self.dataset)
        frame_ids = list(range(n_frames))
        self.mapping_frame_ids = frame_ids[::config["mapping"]["map_every"]] + [n_frames - 1]

        self.estimated_c2ws = torch.empty(len(self.dataset), 4, 4)
        self.estimated_c2ws[0] = torch.from_numpy(self.dataset[0][3])

        save_dict_to_yaml(config, "config.yaml", directory=self.output_path)

        self.submap_using_motion_heuristic = config["mapping"]["submap_using_motion_heuristic"]

        self.keyframes_info = {}
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))

        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids = [0]
        else:
            self.new_submap_frame_ids = frame_ids[::config["mapping"]["new_submap_every"]] + [n_frames - 1]
            self.new_submap_frame_ids.pop(0)

        self.logger = Logger(self.output_path, config["use_wandb"])
        self.mapper = Mapper(config["mapping"], self.dataset, self.logger)
        self.tracker = Tracker(config["tracking"], self.dataset, self.logger)

        # Lightweight progress diagnostic. The normal evaluator computes ATE
        # only after SLAM has finished, which makes a long unstable run costly
        # to discover. GSLAM_ATE_EVERY=N reports the prefix trajectory every N
        # completed frames without writing plots or checkpoints. It uses the
        # same rigid (no-scale) alignment as the final trajectory evaluator.
        try:
            self.ate_every = int(os.environ.get("GSLAM_ATE_EVERY", "0") or 0)
        except ValueError as exc:
            raise ValueError("GSLAM_ATE_EVERY must be a non-negative integer") from exc
        if self.ate_every < 0:
            raise ValueError("GSLAM_ATE_EVERY must be a non-negative integer")
        if self.ate_every:
            print(f"[PeriodicATE] enabled every {self.ate_every} completed frames "
                  "(plot-free; reports raw and aligned RMSE)")

        print('Tracking config')
        pprint.PrettyPrinter().pprint(config["tracking"])
        print('Mapping config')
        pprint.PrettyPrinter().pprint(config["mapping"])

    def _setup_output_path(self, config: dict) -> None:
        """ Sets up the output path for saving results based on the provided configuration. If the output path is not
        specified in the configuration, it creates a new directory with a timestamp.
        Args:
            config: A dictionary containing the experiment configuration including data and output path information.
        """
        if "output_path" not in config["data"]:
            output_path = Path(config["data"]["output_path"])
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = output_path / self.timestamp
        else:
            self.output_path = Path(config["data"]["output_path"])
        self.output_path.mkdir(exist_ok=True, parents=True)
        os.makedirs(self.output_path / "mapping_vis", exist_ok=True)
        os.makedirs(self.output_path / "tracking_vis", exist_ok=True)

    def should_start_new_submap(self, frame_id: int) -> bool:
        """ Determines whether a new submap should be started based on the motion heuristic or specific frame IDs.
        Args:
            frame_id: The ID of the current frame being processed.
        Returns:
            A boolean indicating whether to start a new submap.
        """
        if self.submap_using_motion_heuristic:
            if exceeds_motion_thresholds(
                self.estimated_c2ws[frame_id], self.estimated_c2ws[self.new_submap_frame_ids[-1]],
                    rot_thre=50, trans_thre=0.5):
                return True
        elif frame_id in self.new_submap_frame_ids:
            return True
        return False

    def _report_periodic_ate(self, frame_id: int) -> None:
        """Print prefix ATE without invoking the heavyweight final evaluator."""
        completed = frame_id + 1
        if not self.ate_every or completed % self.ate_every:
            return

        estimated = self.estimated_c2ws[:completed].detach().cpu().numpy()
        gt = np.asarray(self.dataset.poses[:completed])
        valid = np.isfinite(gt).all(axis=(1, 2))
        estimated = estimated[valid]
        gt = gt[valid]
        if not len(gt):
            print(f"[PeriodicATE] {completed} frames: no valid GT poses", flush=True)
            return

        estimated_t = estimated[:, :3, 3]
        gt_t = gt[:, :3, 3]
        raw_rmse = float(np.sqrt(np.mean(np.sum((estimated_t - gt_t) ** 2, axis=1))))

        # Horn alignment, matching src/evaluation/evaluate_trajectory.py.
        estimated_zero = estimated_t - estimated_t.mean(axis=0)
        gt_zero = gt_t - gt_t.mean(axis=0)
        u, _, vh = np.linalg.svd((estimated_zero.T @ gt_zero).T)
        correction = np.eye(3)
        if np.linalg.det(u) * np.linalg.det(vh) < 0:
            correction[2, 2] = -1
        rotation = u @ correction @ vh
        translation = gt_t.mean(axis=0) - rotation @ estimated_t.mean(axis=0)
        aligned_t = (rotation @ estimated_t.T).T + translation
        aligned_rmse = float(np.sqrt(np.mean(np.sum((aligned_t - gt_t) ** 2, axis=1))))

        print(f"[PeriodicATE] {completed} frames (last={frame_id}, valid={len(gt)}): "
              f"raw={raw_rmse * 100:.2f} cm aligned={aligned_rmse * 100:.2f} cm",
              flush=True)

    def start_new_submap(self, frame_id: int, gaussian_model: GaussianModel) -> None:
        """ Initializes a new submap, saving the current submap's checkpoint and resetting the Gaussian model.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        Args:
            frame_id: The ID of the current frame at which the new submap is started.
            gaussian_model: The current GaussianModel instance to capture and reset for the new submap.
        Returns:
            A new, reset GaussianModel instance for the new submap.
        """
        gaussian_params = gaussian_model.capture_dict()
        submap_ckpt_name = str(self.submap_id).zfill(6)
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": sorted(list(self.keyframes_info.keys()))
        }
        save_dict_to_ckpt(
            submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.mapper.keyframes = []
        self.keyframes_info = {}
        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids.append(frame_id)
            self.mapping_frame_ids.append(frame_id)
        self.submap_id += 1
        return gaussian_model

    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        setup_seed(self.config["seed"])
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.submap_id = 0

        # Wall-clock accounting for the mlsys CSV (same definition as the other
        # gaussianslam repos): time around tracker.track() minus visual-odometry
        # time, and time around mapper.map().
        tracking_time_sum = 0.0
        odometer_time_sum = 0.0
        mapping_time_sum = 0.0
        mapping_frame_count = 0

        for frame_id in range(len(self.dataset)):

            if frame_id in [0, 1]:
                estimated_c2w = self.dataset[frame_id][-1]
            else:
                t0 = time.time()
                odo0 = self.tracker.odometer.elapsed
                estimated_c2w = self.tracker.track(
                    frame_id, gaussian_model,
                    torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))
                # Visual-odometry time is excluded from tracking time.
                odo = self.tracker.odometer.elapsed - odo0
                odometer_time_sum += odo
                tracking_time_sum += time.time() - t0 - odo
            self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)
            self._report_periodic_ate(frame_id)

            # Reinitialize gaussian model for new segment
            if self.should_start_new_submap(frame_id):
                save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
                gaussian_model = self.start_new_submap(frame_id, gaussian_model)

            if frame_id in self.mapping_frame_ids:
                if (self.mapper.mapping_log
                        or os.environ.get("GSLAM_SKIP_VIS", "0") != "1"):
                    print("\nMapping frame", frame_id)
                gaussian_model.training_setup(self.opt)
                estimate_c2w = torch2np(self.estimated_c2ws[frame_id])
                new_submap = not bool(self.keyframes_info)
                # DIAGNOSTIC (2026-09-29): the mapping-budget sweep's summary
                # printed exactly new_submap_iterations (100) as iters/frame
                # on EVERY cell regardless of MAP_ITERS, which only happens
                # if new_submap is True for every mapped frame - checking
                # that directly rather than guessing further from code
                # reading. self._diag_new_submap_count / _diag_mapped_count
                # settle it: if the ratio is ~1.0, every frame really is
                # getting treated as a new submap (submap_using_motion_heuristic
                # firing every frame, or keyframes_info never persisting);
                # if it's the expected ~1/new_submap_every, this diagnostic
                # is a dead end and the bug is elsewhere.
                self._diag_mapped_count = getattr(self, "_diag_mapped_count", 0) + 1
                self._diag_new_submap_count = getattr(self, "_diag_new_submap_count", 0) + int(new_submap)
                t0 = time.time()
                opt_dict = self.mapper.map(frame_id, estimate_c2w, gaussian_model, new_submap)
                mapping_time_sum += time.time() - t0
                mapping_frame_count += 1

                # Keyframes info update
                self.keyframes_info[frame_id] = {
                    "keyframe_id": len(self.keyframes_info.keys()),
                    "opt_dict": opt_dict
                }
        save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
        total_time = tracking_time_sum + mapping_time_sum
        with open(self.output_path / "fps_metrics.json", "w") as f:
            json.dump({
                "tracking_time_s": tracking_time_sum,
                "odometer_time_s": odometer_time_sum,
                "mapping_time_s": mapping_time_sum,
                "total_time_s": total_time,
                "num_frames": len(self.dataset),
                "num_mapping_frames": mapping_frame_count,
                "overall_fps": len(self.dataset) / max(total_time, 1e-6),
                # exact iteration totals from the fork's own counters
                "tracking_iters_total": int(self.tracker._tracking_iters_total),
                "mapping_iters_total": int(self.mapper._mapping_iters_total),
            }, f, indent=2)
        # GSLAM_SKIP_VIS also quiets everything here EXCEPT tracking_summary()
        # and mapping_summary(): those two carry Gradient reuse/adaptive and
        # Iterations per frame, the lines an experiment actually reads, mixed
        # in with their own noisier sections - trimmed INSIDE tracking_summary
        # instead of dropping the whole call. Everything below this comment
        # is a diagnostic for a DIFFERENT question than "how did reuse do" -
        # new-submap bookkeeping, adaptive-mapping reference fitting, early-
        # stop state, CUDA graph capture stats, binning capacity, ES sync
        # cadence - all genuinely useful, just not for this purpose.
        if (self.mapper.mapping_log
                or os.environ.get("GSLAM_SKIP_VIS", "0") != "1"):
            print(f"[Diag] new_submap flagged {getattr(self, '_diag_new_submap_count', 0)}/"
                  f"{getattr(self, '_diag_mapped_count', 0)} mapped frames "
                  f"(submap_using_motion_heuristic={self.submap_using_motion_heuristic})")
            print(self.mapper.adaptive_mapper.summary())
        if os.environ.get("GSLAM_SKIP_VIS", "0") != "1":
            print(self.tracker.stopper.summary())
        print(self.tracker.tracking_summary())
        print(self.mapper.mapping_summary())
        if os.environ.get("GSLAM_SKIP_VIS", "0") != "1":
            print(self.tracker.iter_graph.summary())
            print(self.tracker.binning_capacity.summary())
            print(self.tracker.es_signals.summary())
        if hasattr(self.dataset, 'summary'):
            # Read `consumer blocked` before believing any prefetch timing:
            # a prefetcher that misses every frame still runs and buys nothing.
            if os.environ.get("GSLAM_SKIP_VIS", "0") != "1":
                print(self.dataset.summary())
            self.dataset.close()
