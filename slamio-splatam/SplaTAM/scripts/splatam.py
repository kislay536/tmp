import argparse
import os
import shutil
import sys
import time
import json
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_BASE_DIR)

sys.path.insert(0, _BASE_DIR)
sys.path.append(_REPO_ROOT)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.prefetch import FramePrefetcher
from utils.grad_variance import GradVarianceProbe
from utils.tile_cost import TileCostProbe
from utils.grad_staleness import GradStalenessProbe
from utils.iter_trace import IterTrace
from utils.map_iter_trace import MapTrace
from utils.grad_reuse import (GradReuse, RenderClock, stash_grads, restore_grads,
                            config_from_env as grad_reuse_env)
# The exact SE(3) log, so the tangent travelled between two iterates is a
# real 6-vector in the tracker's own [rho | theta] order rather than a
# small-angle guess. Only the staleness probe uses it, and only on probe
# iterations, so its host sync costs a run that is already not timed.
from utils.windowed_convergence import _se3_log_numpy
from utils.grad_needs import colour_grad_is_dead
from utils.gn_tracking import (GNTrackingProbe, se3_exp as gn_se3_exp,
                               mat_to_quat as gn_mat_to_quat)
from utils.eval_helpers import report_loss, report_progress, eval, invert_gt_pose
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify, remove_points
from utils.adaptive_mapper import AdaptiveMapper
from utils.early_stop import EarlyStop
from utils.adam_step_probe import AdamStepProbe
from utils.pose_preconditioner import (PosePreconditioner, se3_adjoint,
                                       mat_to_quat_capturable,
                                       quat_trans_grad_to_tangent,
                                       per_gaussian_tangent_grads,
                                       sample_metric, effective_samples,
                                       rank1_alignment,
                                       apply_tangent_step,
                                       tangent_of_pose_delta,
                                       tangent_of_pose_delta_batch,
                                       adaptive_loss_improved,
                                       adaptive_failure_streak)
from utils.adaptive_pruning import AdaptivePruner
from utils.precond_probe import PrecondProbe
from utils.preconditioner_excursion import FirstExcursionRecorder
from utils.pixel_sample import build_tile_mask, is_sparse_phase, PixelSampleTracker
from utils.tracking_cuda_graph import TrackingStepGraph
from utils.binning_capacity import BinningCapacity
from utils.tracking_iteration_graph import TrackingIterationGraph
from utils.es_signals import ESSignalBuffer
from utils.windowed_convergence import (
    AutoIncumbentEnergyConvergence,
    IncumbentConvergence,
    WindowedConvergenceSweep,
    make_windowed_convergence,
)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import build_tracking_optimizer


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking, capturable=False):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return build_tracking_optimizer(param_groups, tracking=True, capturable=capturable)
    else:
        # The parameter was in the signature but this branch silently dropped
        # it, so a mapping optimizer could never be captured: Adam raises
        # "Attempting CUDA graph capture of step() for an instance of Adam but
        # param_groups' capturable is False" the moment a capture starts. The
        # same class of mistake - a capturable flag keyed to the wrong thing -
        # cost a Replica run on the tracking side (ladder caveat 2). Inert at
        # the default; nothing passes capturable=True here yet.
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15, capturable=capturable)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False,
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False,
             tracking_iteration=None, tile_mask=None, mask_multiply=False,
             binning_kwargs=None, pose_samples=None):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
        if pose_samples is not None:
            # THE SAMPLE METRIC needs the per-Gaussian gradient the pose
            # produces, and in ISOTROPIC mode this tensor is the only route the
            # pose has to the loss - transform_to_frame leaves unnorm_rotations
            # untouched when log_scales.shape[1] == 1, so nothing else carries
            # a camera derivative. retain_grad on a non-leaf costs one stored
            # [N,3] and no extra kernels.
            #
            # ANISOTROPIC WOULD BE SILENTLY WRONG, not loudly: the rotation
            # branch gives the pose a second path, transformed_pts.grad would
            # then be an INCOMPLETE decomposition, and the sum invariant that
            # group 14 asserts would quietly stop holding. Refuse instead.
            if params['log_scales'].shape[1] != 1:
                raise ValueError(
                    "preconditioner sample_metric requires isotropic Gaussians: "
                    "anisotropic mode routes the pose through unnorm_rotations "
                    "as well, so transformed_pts.grad is not the whole gradient")
            transformed_gaussians['means3D'].retain_grad()
            pose_samples['pts'] = transformed_gaussians['means3D']
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    _bk = binning_kwargs or {}
    # dL_dcolors is DEAD in this render during tracking: colors_precomp is
    # params['rgb_colors'], a frozen leaf Parameter the pose cannot reach. That
    # is 3 of the ~9 atomics per contribution, and the accumulation as a whole
    # measured 52.4% of renderCUDABackward (DGR_STUB_GRADS ceiling probe).
    # The rule is checked, not assumed - see utils/grad_needs - because the
    # depth/silhouette render three lines below does NOT qualify.
    # The speedup breakdown can restore the stock full backward while keeping
    # every other tracking setting fixed.  Read this here rather than at module
    # import: the experiment config is loaded only after this module, and that
    # config installs the internal breakdown switch before rgbd_slam starts.
    _tracking_only_backward = (
        tracking
        and os.environ.get(
            "_SLAMIO_SPLATAM_TRACKING_ONLY_BACKWARD", "1"
        ) not in ("0", "", "false", "False")
    )
    _skip_cgrad = (
        colour_grad_is_dead(rendervar['colors_precomp'], tracking)
        if _tracking_only_backward else False
    )
    im, radius, *_ = Renderer(raster_settings=curr_data['cam'])(
        **rendervar, tracking_only=_tracking_only_backward,
        skip_color_grad=_skip_cgrad, tile_mask=tile_mask, **_bk)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    # NOT skippable: these "colours" are depth and silhouette computed from the
    # POSE-TRANSFORMED means, so dL_dcolors carries the pose gradient. The
    # helper returns False here on the grad_fn check - the same call that
    # returns True above - which is the whole reason the rule is evaluated per
    # render rather than set from a tracking flag.
    _skip_cgrad_ds = (
        colour_grad_is_dead(depth_sil_rendervar['colors_precomp'], tracking)
        if _tracking_only_backward else False
    )
    depth_sil, *_ = Renderer(raster_settings=curr_data['cam'])(
        **depth_sil_rendervar, tracking_only=_tracking_only_backward,
        skip_color_grad=_skip_cgrad_ds, tile_mask=tile_mask, **_bk)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    #
    # mask_multiply replaces `x[mask].sum()` with `torch.where(mask, x, 0).sum()`
    # on the tracking paths. Boolean-mask indexing is masked_select, whose output
    # size is data-dependent, so PyTorch has to read the mask's element count
    # back to the host to allocate the result - a device-to-host sync, twice per
    # iteration, in the middle of the forward pass. where() keeps the shape
    # static: no sync, no nonzero, no gather. Masked-out positions contribute
    # exactly 0 to the sum, and the gradient is sign(diff) where the mask is set
    # and 0 elsewhere - the same zero pattern masked_select's backward scatter
    # produces.
    #
    # where() rather than `x * mask`, which was the first attempt: NaN * 0 is
    # NaN, not 0, so a multiply lets a masked-out NaN poison the entire sum.
    # That matters here because `mask` contains a nan_mask term specifically to
    # exclude NaN depth/uncertainty pixels - indexing dropped them, a multiply
    # would not. where() selects, so the unmasked branch is never evaluated into
    # the result and NaN/inf there cannot leak.
    #
    # It costs extra FLOPs (every pixel is processed, not just the selected
    # ones), which is the right trade here: the GPU measured only ~40% busy, so
    # idle GPU cycles are cheaper than a host stall.
    #
    # A naive where() rewrite of the .mean() branch below is NOT equivalent - a
    # mean over selected elements differs from a mean over all elements
    # including zeros - but dividing by the selected COUNT is, and mask.sum()
    # is a device scalar, so it stays sync-free. That is what the non-tracking
    # branch does when mask_multiply is set; it is NOT bit-identical to
    # indexing, because it sums H*W terms rather than the selected ones and
    # floating-point addition is not associative. Hence the flag: on the
    # mapping path this perturbs the loss that builds the map, so it has to be
    # A/B-able rather than silently on. The empty-mask case matches too - both
    # forms give nan.
    if use_l1:
        mask = mask.detach()
        if tracking and mask_multiply:
            _err = torch.abs(curr_data['depth'] - depth)
            losses['depth'] = torch.where(mask, _err, torch.zeros_like(_err)).sum()
        elif tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        elif mask_multiply:
            _err = torch.abs(curr_data['depth'] - depth)
            losses['depth'] = (torch.where(mask, _err, torch.zeros_like(_err)).sum()
                               / mask.sum())
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()

    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        if mask_multiply:
            # mask is (1, H, W) and the error is (3, H, W), so broadcasting does
            # what torch.tile did explicitly - one fewer allocation per call.
            _err = torch.abs(curr_data['im'] - im)
            losses['im'] = torch.where(mask, _err, torch.zeros_like(_err)).sum()
        else:
            color_mask = torch.tile(mask, (3, 1, 1))
            color_mask = color_mask.detach()
            losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    # Boolean-mask ASSIGNMENT is index_put_ with a bool mask: the number of
    # elements written is data-dependent, so it needs the mask's nonzero count
    # on the host. That is a sync, and it is outright illegal while a stream is
    # capturing - it was the first thing CUDA graph capture of a whole tracking
    # iteration tripped over.
    #
    # torch.where computes the same result with a static shape: entries where
    # seen is false keep their previous value, exactly as a masked assignment
    # leaves them untouched. Verified exactly equal on random trials.
    #
    # UNCONDITIONAL as of the mapping-side sync audit. It used to be gated on
    # mask_multiply "since it is the same class of change", which meant the
    # MAPPING call site - which never passes that flag - still ran the
    # index_put_ and paid the sync 30 times per frame on TUM and 60 on Replica.
    # Unlike the loss rewrites this one is exactly equal, not equal-up-to-
    # summation-order, so there is nothing to gate: no config can want the
    # syncing version.
    variables['max_2D_radius'] = torch.where(
        seen, torch.max(radius, variables['max_2D_radius']),
        variables['max_2D_radius'])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, *_ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    # Frame prefetch (utils/prefetch.py, shared with MonoGS and Gaussian-SLAM).
    # Default OFF.
    #
    # The datasets are built on CPU when it is on, because the worker thread
    # must stay CUDA-free - SplaTAM captures a tracking iteration on EVERY
    # frame, and in cudaStreamCaptureModeGlobal any thread launching into a
    # non-capturing stream is an error. The host-to-device move goes in via
    # to_device, which runs on the CONSUMER thread. See the contract at the top
    # of utils/prefetch.py.
    #
    # to_device rather than editing call sites: SplaTAM reads its three
    # datasets from seven places, and a missed one is a CPU tensor reaching the
    # rasterizer.
    #
    # recent=2 because dataset[time_idx] is read from two separate places in a
    # frame; against a forward-only queue the second read resyncs every time.
    _pf_cfg = config.get('prefetch', {})
    _pf_on = bool(_pf_cfg.get('enabled', False))
    _ds_device = "cpu" if _pf_on else device

    def _pf_to_device(sample):
        return tuple(x.to(device) if torch.is_tensor(x) else x for x in sample)

    def _pf_wrap(ds, name):
        # With prefetch off the dataset is built on `device` as before and is
        # returned untouched, so this is a no-op rather than a quiet rewrite.
        if not _pf_on:
            return ds
        return FramePrefetcher(ds, {"recent": 2, **_pf_cfg}, name=name,
                               to_device=_pf_to_device)

    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=_ds_device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    dataset = _pf_wrap(dataset, "splatam")
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=_ds_device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        densify_dataset = _pf_wrap(densify_dataset, "densify")
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset,
                                                                        gaussian_distribution=config['gaussian_distribution'])                                                                                                                  
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'],
                                                                                        gaussian_distribution=config['gaussian_distribution'])
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=_ds_device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_dataset = _pf_wrap(tracking_dataset, "tracking")
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    # Wall clock for the SLAM phase only. The harness's own timer wraps the
    # whole process, so it also includes the final evaluation pass and any
    # dataset/CUDA start-up - which has repeatedly swamped real differences
    # (contaminated runs showed ~190s outside the tracking and mapping loops
    # against ~67s on a clean one). This figure stops before evaluation.
    slam_start_time = time.time()
    # COMMIT_AT_LOSS=2 bookkeeping. REQUESTED vs HAPPENED: if the final pose
    # almost never wins, level 2 is level 1 wearing a different tag, and the
    # comparison between them is measuring nothing.
    _final_pose_wins = 0
    _final_pose_scored = 0
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Load Checkpoint
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = invert_gt_pose(gt_pose, time_idx)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0
    
    # Iterate over Scan
    adaptive_mapper = AdaptiveMapper(config.get('adaptive_mapping', {}))
    stopper = EarlyStop(config['tracking'].get('early_stop', {}))
    adaptive_pruner = AdaptivePruner(config['mapping'].get('adaptive_pruning', {}))
    pixel_sampler = PixelSampleTracker(config['tracking'].get('pixel_sample', {}))
    grad_var_probe = GradVarianceProbe(config['tracking'].get('grad_variance', {}))
    tile_cost_probe = TileCostProbe(config['tracking'].get('tile_cost', {}))
    stale_probe = GradStalenessProbe(config['tracking'].get('grad_staleness', {}))
    iter_trace = IterTrace(config['tracking'].get('iter_trace', {}))
    map_trace = MapTrace(config['mapping'].get('iter_trace', {}))
    _gr_cfg = grad_reuse_env(config['tracking'].get('grad_reuse', {}))
    _gr_calibration_frames = int(
        os.environ.get("GRAD_REUSE_CALIBRATE_FRAMES", "0")
    )
    if _gr_calibration_frames < 0:
        raise ValueError("GRAD_REUSE_CALIBRATE_FRAMES must be >= 0")
    if _gr_calibration_frames:
        if not _gr_cfg.get("enabled", False):
            raise ValueError(
                "GRAD_REUSE_CALIBRATE_FRAMES requires GRAD_REUSE >= 2; "
                "otherwise the calibration would spend its run-in and then "
                "have no reuse mechanism to enable."
            )
        _gr_diag = int(
            config['tracking'].get('preconditioner', {}).get(
                'diag_after', 0
            ) or 0
        )
        if _gr_diag <= 0:
            raise ValueError(
                "automatic gradient-reuse calibration needs a fixed positive "
                "preconditioner diag_after (PRE_DIAG_AFTER)."
            )
        if "GRAD_REUSE_WARMUP" in os.environ:
            # EXPLICIT OVERRIDE OF THE DEFAULT DIAG-TIED WARMUP. The default
            # (warmup == diag_after) exists so the full-matrix acquisition
            # never sees a reused gradient before the diagonal-to-full
            # transition. A fixed, smaller warmup is a deliberate hypothesis
            # test against that default - that per-frame gradient
            # correlation is already high a few iterations in, well before
            # diag_after, so protecting the whole diagonal tail is
            # unnecessary. Not validated; compare ATE against the diag-tied
            # default before trusting it.
            #
            # WHY THIS MATTERS WITH ADAPTIVE (GRAD_REUSE_ADAPTIVE=1) TOO,
            # not just the fixed/calibrated window: should_reuse() gates on
            # `iter_idx < self.warmup` BEFORE it ever looks at the adaptive
            # boundary or local_trust - so a DIAG-tied warmup of 20 against a
            # windowed stopper that typically commits around iteration
            # 24-25 (W_PAT=2's own shipped pattern) leaves only a ~4-5
            # iteration gap for the trust-checked boundary to extend into,
            # regardless of how often the cosine check actually passes.
            # Measured: trust_cos=0.95 extended 67.7% of checks, but only
            # 3.6% of iterations were actually reused, because almost none
            # of a typical ~25-iteration frame falls after warmup=20.
            _warmup = int(os.environ["GRAD_REUSE_WARMUP"])
            print(
                "[GradReuse] calibrated mode: explicit "
                f"GRAD_REUSE_WARMUP={_warmup} overrides the default "
                f"DIAG={_gr_diag} transition",
                flush=True,
            )
        else:
            _warmup = _gr_diag
        if "GRAD_REUSE_COOLDOWN" in os.environ:
            print(
                "[GradReuse] calibrated mode ignores fixed "
                f"GRAD_REUSE_COOLDOWN={os.environ['GRAD_REUSE_COOLDOWN']}; "
                "the no-reuse run-in selects the window",
                flush=True,
            )
        _gr_short = os.environ.get(
            "GRAD_REUSE_CALIBRATE_SHORT", "1"
        ) not in ("0", "", "false", "False")
        # The first N tracked frames run with no reuse.  Long frames use the
        # same stop-minus-margin rule as MonoGS/GSLAM; SplaTAM's usual 25-28
        # iteration horizon selects the utility's fixed fractional window
        # instead (25 -> w6/c20, 28 -> w6/c22).
        _gr_cfg.update({
            "warmup": _warmup,
            "cooldown": 0,
            "calibration_frames": _gr_calibration_frames,
            "calibration_margin": int(os.environ.get(
                "GRAD_REUSE_CALIBRATE_MARGIN", "15")),
            "calibration_round": int(os.environ.get(
                "GRAD_REUSE_CALIBRATE_ROUND", "10")),
            "calibration_stat": os.environ.get(
                "GRAD_REUSE_CALIBRATE_STAT", "min"),
            "calibration_exclude_reasons": ("cap",),
            "calibration_short_horizon": _gr_short,
            # STILL THE DIAG TRANSITION, not _warmup - this says where the
            # structural acquisition boundary actually is, for the
            # short-horizon fallback's own "is there enough room after the
            # normal warmup for 2 trust decisions" decision. An explicit
            # GRAD_REUSE_WARMUP override is a hypothesis about the reuse
            # window itself, not a claim that DIAG moved.
            "calibration_phase_boundary": _gr_diag,
            # Floor on the CALIBRATED warmup, in periods (so x2 for
            # GRAD_REUSE=2's raw-iteration count) - only reached by the
            # short-horizon fallback's own fit, not by a plain
            # GRAD_REUSE_WARMUP (which now sets warmup directly, above).
            # Default 3 periods protects the first 6 iterations; raise it to
            # make the reuse window start later regardless of what
            # calibration alone would have picked.
            "calibration_short_min_warmup_periods": int(os.environ.get(
                "GRAD_REUSE_CALIBRATE_MIN_WARMUP_PERIODS", "3")),
        })
        print(
            "[Tracking] Gradient reuse: calibration armed; reuse disabled "
            f"for the first {_gr_calibration_frames} tracked frames, "
            f"warmup={_warmup}, short_horizon={int(_gr_short)}",
            flush=True,
        )
    grad_reuse = GradReuse(_gr_cfg)
    # Internal ablation control.  The public benchmark has one cumulative
    # --keep stage rather than a second reuse-check flag.  When application is
    # off, should_reuse(), trust-signal production and lease consumption still
    # run exactly where they do in the applying arm; only the render bypass is
    # suppressed.
    _gr_apply_reuse = bool(
        config['tracking'].get('_apply_grad_reuse', True)
    )
    if grad_reuse.enabled and config['tracking'].get('cuda_graph', {}).get('enabled'):
        # STILL REFUSED FOR THE OUTER, WHOLE-FRAME GRAPH, and the reason
        # is not the same as it was for the iteration graph. cuda_graph
        # captures the ENTIRE tracking loop as one region, so the reuse
        # decision - a host branch, taken per iteration - would be inside
        # the capture. There is no caller left to hand a bypass flag to.
        #
        # The ITERATION graph is different: it captures one iteration at a
        # time, and iter_graph.run(bypass=...) lets the reuse iterations
        # run eager while the rendering ones replay. That one composes.
        raise ValueError(
            'grad_reuse cannot run with the whole-frame cuda_graph: the '
            'reuse decision is a host branch and it would fall inside the '
            'capture. Use iteration_graph, which takes bypass=.')
    gn_probe = GNTrackingProbe(config['tracking'].get('gn_tracking', {}))
    # Constructed ONCE for the whole run, not per frame: M is carried across
    # frames and that carry is the point of the method. A per-frame object
    # would be a cold start every time, which is the configuration the method
    # is arguing against.
    _pre_cfg = dict(config['tracking'].get('preconditioner', {}))
    _pre_cfg_stop = {k: _pre_cfg.pop(k) for k in
                     ('stop_rel_grad', 'stop_min_iters',
                      'stop_check_every') if k in _pre_cfg}
    _pre_transport = bool(_pre_cfg.pop('transport', False))
    _excursion_cfg = dict(_pre_cfg.pop('excursion_trace', {}))
    # PHASE-SPLIT TRACKING. The preconditioner runs for the first N iterations
    # of each frame, then Adam finishes.
    #
    # THE ARGUMENT. M ~ E[gg^T] scales as g^2 and m as g, so M^-1/2 m is
    # EXACTLY scale-free - the step stays ~lr at the optimum forever, which is
    # why pose_eps can never fire and ships as 0.0. That is a structural reason
    # the preconditioner may be good at ACQUIRING a basin and indifferent at
    # REFINING inside one, and it is not something more iterations can fix.
    # Adam is 0/10 from a predicted pose in this config, but refinement from an
    # already-good pose is a different task and has never been measured.
    #
    # 0 disables. The comparison that means anything is at EQUAL TOTAL RENDERS:
    # handoff=10 with ITERS=20 against ITERS=20 with no handoff.
    _pre_handoff = int(_pre_cfg.pop('handoff', 0))
    # SHADOW-CALIBRATION STATE, RUN-LEVEL because the evidence accumulates
    # across frames and the install happens once. Read from its own config key
    # rather than the preconditioner block: the observation is a function of
    # the per-iteration loss alone, so PosePreconditioner needs no new
    # argument and its validated constructor is untouched.
    _diag_shadow_cfg = dict(config['tracking'].get('diag_shadow', {}) or {})
    _sh_samples = []
    _sh_censored = 0
    _sh_installed = False
    if _diag_shadow_cfg.get("enabled", False):
        print(f"[Preconditioner] SHADOW diagonal calibration: measuring over "
              f"{_diag_shadow_cfg['calibration_frames']} frames at patience "
              f"{_diag_shadow_cfg.get('patience', 1)} while diag_after="
              f"{_pre_cfg.get('diag_after', 0)} runs for real. Nothing is "
              f"rolled back and no frame is tracked with an unvalidated "
              f"transition.", flush=True)
    _pre_stop_ref = str(_pre_cfg.pop('stop_ref', 'frame'))
    # 'rel'  - threshold on the instantaneous |g|/|g_0|  (the shipped rule)
    # 'best' - patience on the BEST |g| seen this frame (monotone, see
    #          PosePreconditioner.stalled)
    _pre_stop_mode = str(_pre_cfg.pop('stop_mode', 'rel'))
    # ONE FULL-RESOLUTION ITERATION AT THE STOPPING MOMENT.
    #
    # THE DEFECT THIS FIXES. is_sparse_phase returns True for
    # full_start <= iter/(num_iters-1) < full_end, and this config runs
    # 0.0 <= frac < 1.0 - so the sparse window covers every iteration EXCEPT
    # THE LAST, and exactly one full-resolution pass happens per frame, at
    # iteration num_iters-1. A frame that STOPS EARLY never reaches it.
    #
    # Confirmed exactly, not inferred: the ES+retune run stopped 288/591 frames
    # early and logged 88671-88368 = 303 unmasked iterations, and 591-288 = 303.
    # The plateau run stopped every frame early and logged 0 unmasked out of
    # 14047 - it never looked at a full image before committing a pose, and came
    # back at ATE 7.96 against 3.0-3.5 for the arms that did.
    #
    # So every stopping arm in the ladder has been silently trading away its
    # dense refinement, and the ones that stop LESS were protected by accident.
    # Running one dense iteration AT the stopping point restores it for ~1 extra
    # render in 24 (4%), independent of where the frame happens to stop.
    _final_dense = bool(_pre_cfg.pop('final_dense', False))
    # One-shot confirmation, the same pattern EarlyStop.check() and the
    # rasterizer's tracking_only flag already use. Without it the only evidence
    # the dense pass fired is the Tile-mask line in the END-OF-RUN summary,
    # which is no help when a full-length run takes ten minutes and the last
    # several were wasted on arms that were silently inert.
    _fd_seen = [False]
    _pre_stop_patience = int(_pre_cfg.pop('stop_patience', 10))
    _pre_stop_every = max(1, int(_pre_cfg_stop.get('stop_check_every', 1)))
    pose_pre = None
    excursion_recorder = None
    # ADAM'S OWN STEP SIZE, WHICH HAS NEVER BEEN MEASURED ON THE CONTROL ARM.
    # The preconditioner prints `step |d| mean` on every run and that number
    # drives the accuracy/step-floor ratio - the one instrument in this record
    # that has transferred across three models. The ratio needs achievable
    # accuracy, which so far has only come from ATE, which needs ground truth.
    # Adam's plateau step is the proposed ground-truth-free substitute, and it
    # is unmeasured because precond0 has no preconditioner object to record it.
    # See utils/adam_step_probe.py for what the comparison decides.
    #
    # Off by default: it adds a per-iteration norm and two copies to the
    # control arm, and the control arm is a timing reference for every speed
    # claim in the ladder. ADAM_STEP_PROBE=1 turns it on.
    adam_probe = None
    if os.environ.get("ADAM_STEP_PROBE", "0") not in ("0", "", "false", "False"):
        adam_probe = AdamStepProbe(device="cuda")
        print("[Adam probe] recording realised SE(3) tangent step per tracking "
              "iteration. Compare the late-band mean against this cell's Adam "
              "ATE in metres - that is the whole test.", flush=True)
    _pre_stop_rel, _pre_stop_min, _pre_stop_every = 0.0, 10, 1
    # STAGE 1 by default. Stage 0 preconditioned the raw 7 numbers and produced
    # a new pathology at every fix - unbounded opening steps, attraction to the
    # gauge direction, the gauge crowding out the trust region - three of which
    # exist only because 7 parameters describe a 6-dimensional object. The
    # tangent has no gauge freedom, so they cannot arise. Set tangent=False to
    # reproduce the Stage 0 behaviour.
    _pre_tangent = bool(_pre_cfg.pop('tangent', True))
    if _pre_cfg.pop('enabled', False):
        _pre_cfg.pop('dim', None)
        # lr comes from the preconditioner block, NOT from tracking.lrs: the
        # two scales are not interchangeable (see the config header and
        # PosePreconditioner's step-calibration comment).
        # AUTO_LR: derive the step magnitude from the config's OWN Adam lrs
        # instead of a hand-set PRE_LR. Adam's step is ~lr_i per coordinate
        # whatever the gradients do, so the norm it would take in the 6-D
        # tangent is sqrt(3 lr_trans^2 + 3 lr_rot^2) - already tuned per
        # dataset by whoever tuned the dataset. TUM gives 4.9e-3 (lrs 1:1),
        # Replica 3.5e-3 (5:1). Carrying TUM's PRE_LR=0.004 to Replica gave
        # 17.49 cm against Adam's 0.24.
        _auto = _pre_cfg.pop('auto_lr', False)
        _tgt = float(_pre_cfg.pop('target_step_norm', 0.0))
        if _auto and _tgt <= 0.0:
            _lr_r = float(config['tracking']['lrs']['cam_unnorm_rots'])
            _lr_t = float(config['tracking']['lrs']['cam_trans'])
            _tgt = math.sqrt(3.0 * _lr_t ** 2 + 3.0 * _lr_r ** 2)
            print(f"[Preconditioner] auto step norm {_tgt:.3e} from "
                  f"cam_trans={_lr_t:g}, cam_unnorm_rots={_lr_r:g}", flush=True)
        # B_0 FROM THIS CONFIG'S OWN TUNED ADAM lrs, same source AUTO_LR uses.
        # rho and theta do not share a scale and B_0 = I asserts they do; the
        # measured lr sweep (opposite-signed dATE/dlr for rot and trans) is that
        # statement as data. Only consumed in bfgs mode.
        if _pre_cfg.pop('bfgs_b0_from_lrs', False):
            _pre_cfg['bfgs_b0'] = (
                float(config['tracking']['lrs']['cam_trans']),
                float(config['tracking']['lrs']['cam_unnorm_rots']))
            print(f"[Preconditioner] BFGS B0 diag from lrs: "
                  f"trans={_pre_cfg['bfgs_b0'][0]:g}, "
                  f"rot={_pre_cfg['bfgs_b0'][1]:g}", flush=True)
        _pre_lr = float(_pre_cfg.pop(
            'lr', config['tracking']['lrs']['cam_trans']))
        _online_lr_tune = os.environ.get(
            "ONLINE_LR_TUNE", "0") not in ("0", "", "false", "False")
        _online_lr_block = int(os.environ.get("ONLINE_LR_BLOCK", "12"))
        _online_lr_halvings = int(
            os.environ.get("ONLINE_LR_MAX_HALVINGS", "4"))
        if _online_lr_block <= 0:
            raise ValueError("ONLINE_LR_BLOCK must be > 0")
        if _online_lr_halvings <= 0:
            raise ValueError("ONLINE_LR_MAX_HALVINGS must be > 0")
        if _online_lr_tune and _pre_handoff > 0:
            raise ValueError(
                "ONLINE_LR_TUNE cannot run with PRE_HANDOFF: refactor() only "
                "counts the preconditioned part of a handoff frame, so its "
                "it/frame signal would be censored")
        if _online_lr_tune:
            _online_base_name = (
                "target_step_norm" if _tgt > 0.0 else "preconditioner lr")
            _online_base_value = _tgt if _tgt > 0.0 else _pre_lr
            print(
                f"[Preconditioner] ONLINE_LR_TUNE=1: tuning "
                f"{_online_base_name} from native k=1 "
                f"(base={_online_base_value:g}) in "
                f"{_online_lr_block}-frame blocks; at most "
                f"{_online_lr_halvings} halvings", flush=True)
        pose_pre = PosePreconditioner(
            dim=(6 if _pre_tangent else 7),   # SE(3) tangent, or 4 quat + 3 trans
            lr=_pre_lr,
            target_step_norm=_tgt,
            online_lr_tune=_online_lr_tune,
            online_lr_block_frames=_online_lr_block,
            online_lr_max_halvings=_online_lr_halvings,
            online_lr_budget=int(config['tracking']['num_iters']),
            online_lr_log_fn=lambda msg: print(msg, flush=True),
            trace_steps=bool(_excursion_cfg),
            device='cuda', **_pre_cfg)
        pose_pre.tangent = _pre_tangent
        if _excursion_cfg:
            excursion_recorder = FirstExcursionRecorder(**_excursion_cfg)
            print(f"[Preconditioner] first-excursion trace armed: "
                  f"{excursion_recorder.path}", flush=True)
        # Gradient-norm stopping: stop when |g| has fallen to this fraction of
        # the frame's own first gradient. 0 disables. _pre_stop_min is a floor
        # in iterations, so a frame that starts near its optimum still takes a
        # few steps before it is allowed to quit.
        _pre_stop_rel = float(_pre_cfg_stop.get("stop_rel_grad", 0.0))
        _pre_stop_min = int(_pre_cfg_stop.get("stop_min_iters", 10))
        if _pre_stop_rel > 0.0 and _pre_stop_mode == "best":
            print(f"[Preconditioner] PLATEAU stopping: no new best |g| for "
                  f"{_pre_stop_patience} iters (improve margin "
                  f"{pose_pre.stop_improve:g}), floor {_pre_stop_min}",
                  flush=True)
        elif _pre_stop_rel > 0.0:
            print(f"[Preconditioner] gradient-norm stopping: |g|/|g0| < "
                  f"{_pre_stop_rel} after >= {_pre_stop_min} iters", flush=True)
        if pose_pre.bfgs and _tgt > 0.0:
            # REFUSE. target_step_norm makes every step exactly `target` long,
            # which is harmless for M (whose step is scale-free anyway) and
            # fatal for BFGS: -B g with B ~ H^-1 is a Newton step that must
            # SHRINK as the gradient shrinks. Pinning its length removes the
            # one property BFGS exists to provide and the optimiser orbits
            # instead of settling. Seen as mean |d| == max |d| == 1.00x Adam
            # over 2490 steps, with ATE 125 cm.
            #
            # BFGS self-scales instead: bfgs_autoscale sets the FIRST step of
            # each frame to lr*sqrt(n) and the secant equation carries the
            # scale from there.
            raise ValueError(
                "preconditioner bfgs=True cannot run with AUTO_LR / "
                "target_step_norm: it fixes every step length and removes "
                "BFGS's self-scaling. Use AUTO_LR=0 with PRE_LR setting the "
                "first-step magnitude.")
        if pose_pre.sample_metric:
            # REFUSE, do not warn. The reduction G^T G runs over N Gaussians and
            # N grows as the map does, so it cannot be captured; a graph would
            # replay the M_inst recorded at capture time forever. That is
            # exactly failure mode 3 from the graph section of the ladder - P
            # frozen at iteration ~4 - and it was SILENT for three runs.
            for _gname in ('cuda_graph', 'iteration_graph'):
                if config['tracking'].get(_gname, {}).get('enabled', False):
                    raise ValueError(
                        f"preconditioner sample_metric cannot run with "
                        f"{_gname}: the per-Gaussian reduction is not "
                        f"fixed-shape, so a capture would freeze the metric")
            print("[Preconditioner] SAMPLE METRIC on: M from the per-Gaussian "
                  "decomposition (G^T G, trace-matched), not rank-1 g g^T",
                  flush=True)
        if _pre_handoff > 0:
            print(f"[Handoff] preconditioner for iters 0-{_pre_handoff - 1}, "
                  f"then Adam. Compare against no-handoff at the SAME total "
                  f"iterations, not against a longer run.", flush=True)
        # C1 AND C2 ARE HANDOFF REPLACEMENTS, SO THE COMBINATION IS REFUSED.
        #
        # Not a style rule - it would be SILENT. Both knobs fire from
        # refactor(), and splatam.py deliberately stops calling refactor() once
        # the Adam phase begins (see the skip below, and why). So a restart or
        # a ramp scheduled at or after the handoff would simply never happen,
        # and the run would produce a perfectly plausible log, a normal ATE,
        # and this arm's tag on a set of numbers identical to the control.
        #
        # That is the sixteen-times-repeated bug of this campaign, and every
        # instance biased a result optimistically. Refuse rather than warn.
        if _pre_handoff > 0 and (pose_pre.restart_at > 0 or pose_pre.ramp_to > 0):
            raise ValueError(
                f"PRE_RESTART/PRE_RAMP cannot run with PRE_HANDOFF "
                f"(={_pre_handoff}): both are driven from refactor(), which is "
                f"not called during the Adam phase, so they would silently "
                f"never fire. They REPLACE the handoff - run with "
                f"PRE_HANDOFF=0.")
        if pose_pre.restart_at > 0:
            print(f"[C1] first-moment restart at within-frame iteration "
                  f"{pose_pre.restart_at}. M and P are kept: this isolates the "
                  f"momentum reset the handoff also performs, WITHOUT the "
                  f"change of update rule. Check 'restarts N/M frames' in the "
                  f"final summary - frames that stop earlier never fire it.",
                  flush=True)
        if pose_pre.ramp_to > 0:
            print(f"[C2] step ramps toward normalised gradient descent over "
                  f"within-frame iterations {pose_pre.ramp_from}->"
                  f"{pose_pre.ramp_to}. One optimiser, one capture per frame. "
                  f"Check 'mean w' in the final summary.", flush=True)
        print(f"[Preconditioner] {pose_pre.summary()}", flush=True)
        # Both graphs capture `optimizer.step()`, which the preconditioner
        # path bypasses entirely - a capture would replay Adam's update
        # forever while the real step happened outside it, and the run would
        # look plausible while measuring nothing. Refuse rather than warn:
        # a silently wrong arm is worse than a config error.
        # THE ITERATION GRAPH IS NO LONGER BLOCKED. It used to be: it captures
        # optimizer.step() as part of the whole iteration, which the
        # preconditioner replaces. But when it is on it DISABLES the step graph
        # and runs the update eagerly inside the outer capture (see the
        # TrackingIterationGraph setup below), and that update is now
        # capture-safe - branch-free helpers, in-place state, device-side bias
        # counter, all verified by profiling/probe_precond_capture.py.
        #
        # Everything that cannot be captured already lives outside
        # _tracking_iteration: refactor() (the eigh) and the stopping check's
        # readback both run after tracking_iteration_graph.run().
        #
        # This is UNVALIDATED on a real run - the probe covers the step alone,
        # not the step nested inside a whole-iteration capture. It is worth
        # trying because the iteration graph is what Replica's own ladder rungs
        # use, and it is worth far more than the step graph (2.71x on the
        # SplaTAM ladder against the step graph's measured net loss).
        if config['tracking'].get('iteration_graph', {}).get('enabled', False):
            print("[Preconditioner] iteration_graph is ON with the "
                  "preconditioner. The update is capture-safe but this "
                  "combination has not been validated end-to-end - check the "
                  "capture summary and that ATE matches a graph-off run.",
                  flush=True)
    # n_eff DIAGNOSTIC, accumulated on device and read ONCE at the end - a
    # per-iteration .item() is a host sync in the hot loop, which is the same
    # trade STOP_EVERY already makes for the stopping check.
    #
    # THIS IS THE NUMBER THAT SAYS WHETHER THE ARM CAN POSSIBLY WORK.
    # n_eff = sum_i |g_i|^2 / |sum_i g_i|^2 is how many INDEPENDENT samples a
    # backward is worth. Near 1 means the per-Gaussian gradients are one
    # gradient in disguise, G^T G is a rescaled g g^T, and no tuning separates
    # the arms - read it before running a sweep, not after.
    _pre_neff = torch.zeros((), device='cuda')
    _pre_neff_n = torch.zeros((), device='cuda')
    _pre_cos = torch.zeros((), device='cuda')
    # MECHANISM, not outcome. ATE at a fixed budget cannot test the
    # sample-efficiency claim - see the header of utils/precond_probe.py.
    # A probe run is NOT a timed run: every row costs a 6x6 eigh and
    # several host syncs inside the tracking loop.
    precond_probe = PrecondProbe(config['tracking'].get('precond_probe', {}))

    def _pre_neff_line():
        if pose_pre is None or not pose_pre.sample_metric:
            return ""
        _n = float(_pre_neff_n)
        if _n <= 0:
            # NOT "no data" - the arm is on and no backward reached it, which
            # means the wiring is broken and the run is measuring the shipped
            # rank-1 method under the new arm's name. Say so in words.
            return "  sample metric: NO BACKWARDS SAMPLED - wiring is broken"
        # cos(G^T G, g g^T) IS THE GO/NO-GO NUMBER, not n_eff. n_eff measures
        # CANCELLATION (~1 incoherent, ~1/N when every Gaussian pulls the same
        # way) and says only that the trace rescaling is real. Whether the
        # metric's SHAPE differs from the rank-1 one is this cosine, and it was
        # briefly - and wrongly - documented as n_eff's job.
        _c = float(_pre_cos) / _n
        return (f"  sample metric: cos(G^T G, g g^T) = {_c:.3f} "
                f"{'<- ~1.0, a rescaled g g^T: STOP' if _c > 0.99 else '(< 1 = new directions, the premise)'}"
                f"   n_eff = {float(_pre_neff) / _n:.2f} "
                f"({int(_n)} backwards)")
    tracking_step_graph = TrackingStepGraph(config['tracking'].get('cuda_graph', {}))
    binning_capacity = BinningCapacity(config['tracking'].get('binning_capacity', {}))
    _iter_graph_cfg = dict(config['tracking'].get('iteration_graph', {}))
    tracking_iteration_graph = TrackingIterationGraph(_iter_graph_cfg)
    if tracking_iteration_graph.enabled:
        # Keyed on the ITERATION graph alone, deliberately. This used to require
        # `and tracking_step_graph.enabled`, which meant keep_grads was only set
        # when cuda_graph happened to be on - and with it off,
        # TrackingStepGraph.step() takes its disabled path and calls
        # zero_grad(set_to_none=True) INSIDE the outer capture, freeing .grad so
        # the next backward allocates at new addresses. That is blocker 4 from
        # the write-up, reintroduced through a config that simply turned off a
        # flag documented as inert.
        if tracking_step_graph.enabled:
            # The iteration graph captures optimizer.step() as part of the whole
            # iteration, so the step-only graph is subsumed. Leaving it on makes
            # it replay its own graph inside the outer capture, which fails with
            # "CUDA generator expects graph capture to be underway".
            print("[TrackingIterationGraph] disabling the step-only cuda_graph: "
                  "the iteration graph already covers optimizer.step()", flush=True)
            tracking_step_graph.enabled = False
        # Its step now runs inside the outer capture, so grads must stay
        # allocated - freeing them would move .grad addresses every iteration.
        tracking_step_graph.keep_grads = True
    _wc = dict(config['tracking'].get('windowed_convergence', {}))
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
        "WCONV_ANOMALY_RELEASE_AFTER": ("anomaly_release_after", int),
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
        "WCONV_AUTO_SKIP": ("auto_calibration_skip_frames", int),
        "WCONV_AUTO_CALIB": ("auto_calibration_frames", int),
        "WCONV_AUTO_AUDIT": ("auto_audit_every", int),
        "WCONV_AUTO_LATE_AUDIT": ("auto_late_audit_frame", int),
        "WCONV_AUTO_LATE_AUDIT_FRAMES": ("auto_late_audit_frames", int),
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
    if pose_pre is not None:
        _wc_coord_scale = (
            float(pose_pre.target_step_norm) / math.sqrt(6.0)
            if float(pose_pre.target_step_norm) > 0.0
            else float(pose_pre.lr)
        )
        _wc_scales = [_wc_coord_scale] * 6
    else:
        # tangent_of_pose_delta returns [rho | theta]. Adam's natural
        # coordinate scales therefore follow translation first, rotation last.
        _wc_scales = (
            [float(config['tracking']['lrs']['cam_trans'])] * 3
            + [float(config['tracking']['lrs']['cam_unnorm_rots'])] * 3
        )
    _wc["budget"] = int(config['tracking']['num_iters'])
    windowed_stop = make_windowed_convergence(_wc, scales=_wc_scales)
    windowed_sweep = WindowedConvergenceSweep(
        os.environ.get("WCONV_SWEEP", ""), _wc, scales=_wc_scales
    )
    # Gradient reuse produces no new loss. Both opt-in clocks therefore drop
    # reused rows from the windowed stopper. `render` conservatively banks the
    # stale-gradient displacement into the next rendered row; `fresh` also
    # drops that displacement from convergence evidence. The optimiser still
    # applies the FULL reuse step either way - only what the stopper observes
    # changes. Candidate selection and the legacy stopper stay untouched.
    _reuse_stop_clock = os.environ.get(
        "GRAD_REUSE_STOP_CLOCK", "").strip().lower()
    if _reuse_stop_clock not in ("", "iteration", "render", "fresh"):
        raise ValueError(
            "GRAD_REUSE_STOP_CLOCK must be iteration, render, or fresh")
    _render_clock = (
        grad_reuse.enabled and windowed_stop.enabled
        and _reuse_stop_clock in ("render", "fresh"))
    if _render_clock:
        if _reuse_stop_clock == "render":
            print("[Tracking] windowed stopper runs on the RENDER clock "
                  "(reused rows skipped; their steps banked)", flush=True)
        else:
            print("[Tracking] windowed stopper runs on FRESH evidence "
                  "(reused rows and their steps omitted; optimiser steps "
                  "remain full scale)", flush=True)
    if grad_reuse.calibration_frames:
        if pose_pre is None:
            raise ValueError(
                "automatic gradient-reuse calibration requires the pose "
                "preconditioner; DIAG is the protected acquisition phase."
            )
        if not windowed_stop.active:
            raise ValueError(
                "automatic gradient-reuse calibration needs active windowed "
                "stopping (WCONV=1, not shadow-only); otherwise its run-in has "
                "no uniform real-stop signal and cap frames are excluded."
            )
    _es_cfg = dict(config['tracking'].get('early_stop', {}).get('batched', {}))
    if windowed_stop.enabled or grad_reuse.batched_check:
        _es_cfg['enabled'] = True
        _es_cfg['batch'] = int(os.environ.get('WCONV_BATCH', '8'))
    es_signals = ESSignalBuffer(
        _es_cfg,
        device=params['cam_trans'].device,
        extra_cols=(grad_reuse.batched_signal_cols
                    if grad_reuse.batched_check else 0),
        include_pose_pair=windowed_stop.enabled,
    )
    if grad_reuse.batched_check and not es_signals.enabled:
        raise ValueError(
            "GRAD_REUSE_BATCHED_CHECK requires the batched stopping signal "
            "buffer (use WCONV=1)")
    if grad_reuse.batched_check:
        print("[Tracking] adaptive reuse trust rides the existing signal "
              f"drain (batch={es_signals.batch}, lease="
              f"{grad_reuse.trust_lease}, period=2; no standalone readback)",
              flush=True)
    if (windowed_stop.active
            and windowed_stop.check_every % es_signals.batch != 0):
        raise ValueError(
            "active windowed_convergence check_every must be a multiple "
            "of es_signals.batch so its decision lands on a drain boundary"
        )
    # The one-shot automatic controller runs full-budget calibration frames,
    # then delegates every active decision to IncumbentEnergyConvergence (an
    # IncumbentConvergence subclass).  It therefore has the same final-dense
    # contract as the already validated fixed rule: a proposal schedules one
    # dense iteration, and the unconditional _force_dense branch commits it.
    _final_dense_compatible = isinstance(
        windowed_stop,
        (IncumbentConvergence, AutoIncumbentEnergyConvergence),
    )
    if (windowed_stop.active and _final_dense
            and not _final_dense_compatible):
        raise ValueError(
            "FINAL_DENSE with active windowed convergence is currently "
            "validated only for incumbent convergence and the one-shot "
            "automatic incumbent-energy decay controller"
        )
    if es_signals.enabled and config['report_iter_progress']:
        # report_progress renders and reads back every iteration, so it syncs
        # anyway - batching would remove three syncs and leave a bigger one.
        # Rather than pretend, refuse: a run that silently measures nothing is
        # worse than one that will not start.
        raise ValueError("early_stop.batched and report_iter_progress are "
                         "incompatible: per-iteration progress reporting syncs "
                         "every iteration, which is exactly what batching removes.")
    print(f"[Tracking] {windowed_stop.summary()}", flush=True)
    for _line in windowed_sweep.summary_lines():
        print(f"[Tracking] {_line}", flush=True)
    # The automatic stopping controller deliberately runs its skipped prefix
    # and calibration frames at the full tracking budget.  Keep those real
    # iterations in the timing totals, but do not let them inflate the
    # steady-state iterations/frame value reported to the results CSV.
    _tracking_report_warmup_frames = (
        int(windowed_stop.calibration_skip_frames)
        + int(windowed_stop.calibration_frames)
        if isinstance(windowed_stop, AutoIncumbentEnergyConvergence)
        else 0
    )
    _tracking_post_warmup_steps = 0
    _tracking_post_warmup_frames = 0
    if tracking_iteration_graph.enabled:
        # Capture is attempted at iteration `warmup_iters`, so a frame that
        # stops before that never captures and silently runs the whole frame
        # eager - no error, no fallback counted, just the speedup quietly gone.
        # min_iters is 70 in every tuned config so there is a wide margin, but
        # retuning can lower it (retune_min_iters_floor defaults to 8 in the
        # EarlyStop class), which would put it in reach.
        _es_floor = min(stopper.min_iters, stopper.warmup_min_iters,
                        stopper.retune_min_iters_floor if stopper.retune_every > 0
                        else stopper.min_iters)
        if stopper.enabled and _es_floor <= tracking_iteration_graph.warmup_iters:
            print(f"[TrackingIterationGraph] WARNING: early stopping can fire at "
                  f"iteration {_es_floor}, at or below warmup_iters="
                  f"{tracking_iteration_graph.warmup_iters}. Frames that stop that "
                  f"early never reach the capture and run fully eager - check the "
                  f"'frames captured' line in the summary.", flush=True)
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = invert_gt_pose(gt_pose, time_idx)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        # Gauss-Newton feasibility probe (STEP 1 of the second-order tracking
        # plan). Fires here because this is the MOTION-MODEL PREDICTION - the
        # exact pose the real tracker starts from - so both arms begin from the
        # same initialisation. Measures only; nothing below sees it.
        #
        # DEPTH-ONLY RESIDUAL, deliberately. It is the best-conditioned choice
        # and closest to ICP, so if Gauss-Newton cannot converge on THIS it
        # cannot converge on the photometric one either, and the idea dies in an
        # hour instead of a week. The colour and combined residuals come after.
        _gn_render_fn = None
        _gn_render_grad_fn = None
        _gn_gt = None
        # Built for probed frames OR on every frame in tracker mode, where the
        # render_fn IS the tracker and a missing one would silently leave the
        # pose at the motion-model prediction for that frame.
        # scan_enabled is the third case and it is NOT covered by the other
        # two: the line-scan runs with GN entirely off, so both should_probe
        # and use_as_tracker are False and the closure would never be built.
        _gn_scan_now = (gn_probe.scan_enabled and time_idx > 0
                        and time_idx % gn_probe.scan_every == 0)
        if (gn_probe.should_probe(time_idx) or gn_probe.use_as_tracker
                or _gn_scan_now):
            _gn_cam = tracking_curr_data['cam']
            _gn_obs_depth = tracking_curr_data['depth'][0]
            _gn_sil = config['tracking']['sil_thres']
            if params['log_scales'].shape[1] != 1:
                print("[GNTrack] anisotropic Gaussians: the prototype does not "
                      "transform rotations, so the render would be wrong. "
                      "Skipping rather than reporting a number from it.",
                      flush=True)
            else:
                # THE RESIDUAL MUST BE THE ONE ADAM MINIMISES, or the two arms
                # are solving different problems and the comparison is void.
                # SplaTAM tracks on loss_weights im=0.5, depth=1.0 - depth-only
                # was tested first and FAILED for a geometric reason: cost fell
                # 2-12x while pose error rose 3-11x, because depth-only
                # alignment on the planar structure of fr1_desk is
                # under-constrained in the tangent directions (the classic ICP
                # degeneracy). Colour is what breaks it.
                #
                # Residuals are scaled by sqrt(weight) so the sum of squares
                # equals the weighted objective, and the Huber deltas are
                # scaled identically so one elementwise comparison is
                # meaningful across metres and intensities.
                _gn_mode = config['tracking']['gn_tracking'].get('residual', 'both')
                _w_im = float(config['tracking']['loss_weights']['im'])
                _w_dp = float(config['tracking']['loss_weights']['depth'])
                _d_dp = float(config['tracking']['gn_tracking'].get('huber_depth', 0.05))
                _d_im = float(config['tracking']['gn_tracking'].get('huber_im', 0.10))
                _gn_obs_im = tracking_curr_data['im']

                def _gn_render(_w2c, _p=params, _cam=_gn_cam, _obs=_gn_obs_depth,
                               _obs_im=_gn_obs_im, _sil=_gn_sil, _mode=_gn_mode,
                               _wi=_w_im, _wd=_w_dp, _di=_d_im, _dd=_d_dp,
                               _grad=False, _w2c0=tracking_curr_data['w2c']):
                    # _grad=True is the SAME function with the no_grad dropped,
                    # so the FD-vs-autograd check compares two evaluations of
                    # one expression rather than of two hand-written variants
                    # that could drift apart. The Gaussians stay detached in
                    # both: only the pose carries gradient.
                    with torch.enable_grad() if _grad else torch.no_grad():
                        _pts = _p['means3D'].detach()
                        _p4 = torch.cat([_pts, torch.ones_like(_pts[:, :1])], 1)
                        _tg = {'means3D': (_w2c @ _p4.t()).t()[:, :3],
                               'unnorm_rotations': _p['unnorm_rotations'].detach()}
                        # DOUBLE-TRANSFORM BUG, fixed. This argument is NOT the
                        # camera pose - transformed_params2depthplussilhouette
                        # feeds it to get_depth_and_silhouette, which applies it
                        # to means3D that are ALREADY in camera frame. SplaTAM
                        # passes curr_data['w2c'] = first_frame_w2c, and the
                        # loaders set relative_pose=True so the first pose is
                        # identity - i.e. SplaTAM is passing identity and the
                        # rendered depth is simply the camera-frame z.
                        # Passing _w2c here instead rendered z of
                        # (w2c @ w2c @ pts): correct 2D positions carrying
                        # wrong depth VALUES, so the depth residual - which
                        # carries weight 1.0 against colour's 0.5 - was
                        # comparing garbage against the observed depth.
                        _rv = transformed_params2depthplussilhouette(_p, _w2c0, _tg)
                        _ds, *_rest = Renderer(raster_settings=_cam)(**_rv)
                        # valid = observed depth present AND the map actually
                        # covers this pixel. Both conditions are what the real
                        # tracking loss uses.
                        _valid = (_obs > 0) & (_ds[1] > _sil)
                        _rd = (_ds[0] - _obs).unsqueeze(0)
                        if _mode == 'depth':
                            return {"residual": _rd, "mask": _valid.unsqueeze(0),
                                    "huber_delta": torch.full_like(_rd, _dd)}
                        _rvc = transformed_params2rendervar(_p, _tg)
                        _im, *_r2 = Renderer(raster_settings=_cam)(**_rvc)
                        _ri = _im - _obs_im                      # (3, H, W)
                        if _mode == 'color':
                            _m = _valid.unsqueeze(0).expand_as(_ri)
                            return {"residual": _ri, "mask": _m,
                                    "huber_delta": torch.full_like(_ri, _di)}
                        _sd, _si = _wd ** 0.5, _wi ** 0.5
                        _res = torch.cat([_sd * _rd, _si * _ri], 0)   # (4, H, W)
                        _msk = _valid.unsqueeze(0).expand_as(_res)
                        _del = torch.cat([torch.full_like(_rd, _sd * _dd),
                                          torch.full_like(_ri, _si * _di)], 0)
                        return {"residual": _res, "mask": _msk,
                                "huber_delta": _del}

                # Stored, not called. The probe now fires INSIDE the tracking
                # loop at each handoff iteration, launching GN from whatever
                # pose Adam has reached by then - so one tracking run yields
                # the whole "GN iterations vs how long Adam ran" curve.
                _gn_render_fn = _gn_render
                _gn_render_grad_fn = (
                    (lambda _w: _gn_render(_w, _grad=True))
                    if config['tracking']['gn_tracking'].get('jac_check')
                    else None)
                _gn_gt = tracking_curr_data['iter_gt_w2c_list'][-1].to(
                    params['means3D'].device).float()

        # Tracking
        tracking_start_time = time.time()
        # Adam runs unless this is PURE GN (handoff 0). In hybrid mode Adam
        # does the non-convex coarse phase, capped at tracker_handoff, and GN
        # refines afterwards - which is the pipeline the handoff sweep was
        # built to measure and which the first tracker mode never implemented.
        if (time_idx > 0 and not config['tracking']['use_gt_poses']
                and not (gn_probe.use_as_tracker
                         and gn_probe.tracker_handoff == 0)):
            # Reset Optimizer & Learning Rates for tracking
            _cuda_graph_cfg = config['tracking'].get('cuda_graph', {})
            # capturable must be on for EITHER graph. It used to be keyed on
            # cuda_graph alone, which made the iteration graph silently depend
            # on a flag the ladder file describes as doing nothing: with
            # cuda_graph off, Adam raises "Attempting CUDA graph capture of
            # step() for an instance of Adam but param_groups' capturable is
            # False" the moment the iteration graph tries to capture. Every TUM
            # ladder config happens to set cuda_graph=True, so this was never
            # hit there - the first config that did not (Replica) failed
            # immediately.
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True,
                                              capturable=(_cuda_graph_cfg.get('enabled', False)
                                                          or tracking_iteration_graph.enabled))
            # Stage 0 of the full-matrix preconditioner: n = 7 on the existing
            # (quaternion, translation) coordinates. Not the principled version
            # - the unnormalised quaternion carries a gauge direction that is an
            # exact null direction of the loss, and there is no SE(3) adjoint in
            # these coordinates, so the cross-frame carry cannot be transported
            # and is passed transport=None. It answers "does a full matrix beat
            # a diagonal at all" without the tangent-space surgery.
            if pose_pre is not None:
                # CROSS-FRAME TRANSPORT. M lives in the tangent at the current
                # pose, and with left perturbation T <- exp(dxi) T that tangent
                # is in CAMERA-frame coordinates. The camera moves between
                # frames, so M_{t-1} and M_t are stated in different coordinate
                # systems and copying the matrix silently mixes them. The
                # adjoint is the correction:
                #
                #     M <- A^-T M A^-1,   A = Ad_{T_t T_{t-1}^-1}
                #
                # Only meaningful in the tangent - Stage 0's quaternion
                # coordinates have no adjoint, which is why this was passed
                # None until Stage 1 landed.
                _A = None
                if (_pre_transport and pose_pre.tangent and time_idx > 0):
                    with torch.no_grad():
                        def _T_of(_i):
                            _T = torch.eye(4, device=params['cam_trans'].device,
                                           dtype=params['cam_trans'].dtype)
                            _T[:3, :3] = build_rotation(F.normalize(
                                params['cam_unnorm_rots'][..., _i].detach()))[0]
                            _T[:3, 3] = params['cam_trans'][..., _i].detach().reshape(3)
                            return _T
                        _A = se3_adjoint(_T_of(time_idx) @ torch.linalg.inv(_T_of(time_idx - 1)))
                pose_pre.carry_frame(transport=_A)
                # AFTER carry_frame, so the shadow estimator starts from the
                # same carried metric the real one does - the curve then
                # measures the two diverging from a common initial
                # condition rather than from an arbitrary one.
                precond_probe.start_frame(time_idx, pose_pre.M)
                # Report the step calibration EARLY. The end-of-run summary is
                # useless for tuning PRE_LR, because a wrong PRE_LR is visible
                # by frame 5 and the run gets killed long before it prints.
                # These few syncs are per-frame, not per-iteration.
                _prof_due = pose_pre.profile_due()
                # Gated on profile_every>0 (PRE_PROFILE_EVERY), not just
                # "always" - these two prints are the bulk of a clean run's
                # output otherwise: the summary's own REL-GRAD/IT-FRAME/K
                # series dump and the step_profile() distribution are both
                # meant for active PRE_LR tuning or profiling/lr_proxy_replay.py,
                # not for reading a finished run.
                if pose_pre.profile_every > 0 and (pose_pre.frames in (5, 20, 50) or _prof_due):
                    print(f"\n[Preconditioner] {pose_pre.summary()}", flush=True)
                    # The pre-cap step distribution, EARLY, for the same reason
                    # the summary is printed early: it is the number that says
                    # whether the metric is overreaching, and waiting for a
                    # full run to find out wastes the run.
                    _sp = pose_pre.step_profile()
                    if _sp:
                        print(_sp, flush=True)
                    # RESET AFTER PRINTING, so the next report is a fresh
                    # window. A cumulative profile over 590 frames averages the
                    # ~20 frames where the tracker breaks into the ~570 where
                    # it does not, which is exactly the signal a full-length
                    # run is being paid for.
                    if _prof_due:
                        pose_pre.reset_profile()
                    # n_eff EARLY, for the same reason. It is the go/no-go
                    # number for this arm and it is knowable by frame 5: if
                    # the per-Gaussian gradients are coherent, n_eff sits
                    # near 1, G^T G is a rescaled g g^T, and the sweep is
                    # not worth running. No point learning that from an
                    # end-of-run line after a ten-minute run.
                    if _pre_neff_line():
                        print(_pre_neff_line(), flush=True)
            tracking_step_graph.reset_for_frame()
            binning_capacity.reset_for_frame()
            tracking_iteration_graph.reset_for_frame()
            es_signals.reset_for_frame()
            windowed_stop.reset_frame()
            windowed_sweep.reset_frame()
            if adam_probe is not None:
                adam_probe.note_frame()
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            # C0. PERSISTENT SNAPSHOT BUFFERS for the Adam phase's step size.
            #
            # ALLOCATED HERE, ONCE PER FRAME, AND WRITTEN WITH copy_ INSIDE THE
            # LOOP. A `.clone()` at the call site would allocate a fresh tensor
            # every iteration, which inside a captured region comes from the
            # graph's private pool and is rebound by Python that never runs on
            # a replay. That is failure mode 1 from the graph section - the one
            # that made rel_grad() read zeros and cut frames after four
            # iterations. Fixed addresses, in-place writes, no rebinding.
            _c0_q = params['cam_unnorm_rots'][..., time_idx].detach().reshape(4).clone()
            _c0_t = params['cam_trans'][..., time_idx].detach().reshape(3).clone()
            current_min_loss = float(1e20)
            # Sync-free candidate selection keeps the running minimum on the
            # GPU so `loss < current_min_loss` never has to be converted to a
            # Python bool - see the update site in the loop below.
            _mask_multiply = config['tracking'].get('mask_multiply_loss', False)
            _sync_free_candidates = config['tracking'].get('sync_free_candidates', False)
            # COMMIT_AT_LOSS. Commit the pose the best loss was evaluated at,
            # instead of the one the optimiser stepped to immediately after.
            # See the note at the candidate update below. Read from the same
            # place es_signals reads it so the batched and eager paths can
            # never disagree about which pose a frame commits.
            _commit_at_loss_level = int(config['tracking'].get('early_stop', {})
                                        .get('batched', {}).get('pose_at_loss', 0))
            _commit_at_loss = _commit_at_loss_level >= 1
            if _sync_free_candidates:
                current_min_loss_t = torch.full((), float('inf'),
                                                device=params['cam_trans'].device,
                                                dtype=params['cam_trans'].dtype)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            _prev_sparse_iter = None
            # Set when a stopping criterion has fired and the frame owes one
            # full-resolution iteration before it commits the pose.
            _force_dense = False
            # Exactly one reason is handed to the reuse calibrator after the
            # frame.  It starts as cap and is replaced only by a rule that
            # genuinely fired; cap is excluded from the calibration statistic.
            _gr_stop_reason = "cap"
            num_iters_tracking = config['tracking']['num_iters']
            if gn_probe.use_as_tracker and gn_probe.tracker_handoff > 0:
                # HYBRID: Adam gets at most this many iterations, then hands
                # over. Early stopping may end it sooner, which is fine - the
                # handoff is an upper bound, not a target.
                num_iters_tracking = gn_probe.tracker_handoff
            # Tile subsampling: build mask once per frame, reuse across
            # sparse-phase iterations (masked tiles write background in
            # forward, backward skips them via the existing n_contrib==0
            # early-exit path - real GPU time saved, not just Python-side
            # loss masking).
            _ps_cfg = config['tracking'].get('pixel_sample', {})
            _tile_mask = None
            if _ps_cfg.get('enabled', False):
                _ps_H = tracking_curr_data['im'].shape[1]
                _ps_W = tracking_curr_data['im'].shape[2]
                _tile_mask = build_tile_mask(tracking_curr_data['im'], _ps_H, _ps_W, _ps_cfg)
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            # NVTX range so a profiler can isolate TRACKING's rasterizer kernels
            # from MAPPING's - they launch the same kernel and no launch property
            # separates them. Mirrors monogs_tracking / gslam_tracking.
            #
            # Push/pop ranges are THREAD-LOCAL and autograd runs backward on its
            # own CUDA worker thread, so this does NOT cover renderCUDABackward
            # unless backward is forced onto the calling thread.
            # SPLATAM_PROFILE_SYNC_AUTOGRAD=1 does that; profiling-only, since it
            # changes scheduling - never set it for a timing run.
            if os.environ.get("SPLATAM_PROFILE_SYNC_AUTOGRAD") == "1" and hasattr(
                torch.autograd, "set_multithreading_enabled"
            ):
                torch.autograd.set_multithreading_enabled(False)
            torch.cuda.nvtx.range_push("splatam_tracking")
            stale_probe.reset_frame()
            iter_trace.begin_frame(time_idx, num_iters_tracking,
                                   params['cam_unnorm_rots'][..., time_idx],
                                   params['cam_trans'][..., time_idx])
            grad_reuse.reset_frame()
            # The stash never crosses a frame boundary - the pose jumps to a
            # new initialisation between frames, so a carried gradient would
            # have been evaluated at a pose the optimiser is nowhere near.
            _gr_stash = [None]
            _gr_last = [None]
            # A LIST, not a closure-local, because the CALLER reads it now:
            # a reuse iteration has to bypass the iteration graph, and
            # iter_graph.run() must be told before it would otherwise
            # replay a capture that renders.
            _gr_reused = [False]
            # Candidate and applied reuse are distinct in the breakdown
            # control.  A one-element list makes both values visible on the
            # caller side of the captured closure without rebinding.
            _gr_candidate = [False]
            # Side channel out of the captured iteration.  The list assignment
            # executes during capture; graph replay rewrites the referenced
            # tensor in place, and the host trust check happens after run().
            _gr_gxi_out = [None]
            _clock = (RenderClock(bank_steps=_reuse_stop_clock == "render")
                      if _render_clock else None)
            stopper.reset()
            pose_delta_norm = 1.0
            # Batched-signal state. The candidate is tracked on the HOST here,
            # from the rows the drain returns, and copied to the device once at
            # the end of the frame. That keeps the comparison in the same plain
            # Python the eager path uses - the on-device torch.where version
            # (sync_free_candidates) broke tracking at ATE 64.28cm and its root
            # cause was never found, so only the FREQUENCY of the host
            # comparison changes here, never where it happens.
            _best_loss_host = float('inf')
            _best_rot_host = _best_tran_host = None

            def _consume(rows):
                """Feed drained iterations to the stopper and the candidate.

                Returns True if early stopping fired. Rows after the firing one
                are dropped: they were executed (the stop is detected up to
                batch-1 iterations late) but the eager path would never have run
                them, so ignoring them keeps the decisions identical and makes
                this a pure performance change.
                """
                nonlocal _best_loss_host, _best_rot_host, _best_tran_host
                nonlocal _gr_stop_reason
                _latest_trust = None
                _trust_vetoed = False
                # ONE BATCHED TANGENT PER DRAIN, NOT ONE CALL PER ROW. The per-row call
                # dispatched a few dozen tiny float64 CPU ops for every counted iteration
                # while the GPU waited: ~1.15-1.33 ms/row on CPU, a large share of why
                # stopping cut 31% of GSLAM/fr1_desk's iterations but ~5% of its tracking
                # time. tangent_of_pose_delta_batch is pinned to the single-row result by
                # utils/test_tangent_batch.py (max difference 0.0 across 1e-5..1 rad), so
                # every stopping decision is unchanged. Rows after a stop are still
                # ignored; computing their tangents up front has no side effects.
                _steps = None
                if windowed_stop.enabled and rows:
                    _steps = tangent_of_pose_delta_batch(
                        torch.stack([_r["pre_rot"] for _r in rows]),
                        torch.stack([_r["pre_tran"] for _r in rows]),
                        torch.stack([_r["post_rot"] for _r in rows]),
                        torch.stack([_r["post_tran"] for _r in rows]),
                    ).detach().cpu().numpy()
                for _j, _r in enumerate(rows):
                    if (grad_reuse.batched_check
                            and _r.get("extra") is not None
                            and _r["extra"][0] == _r["extra"][0]):
                        # Several fresh pairs may land in one real-iteration
                        # batch. The newest exact pair best predicts the NEXT
                        # block; older readings authorize nothing retroactively.
                        _latest_trust = (
                            _r["extra"][0],
                            (_r["extra"][1]
                             if len(_r["extra"]) > 1 else None),
                            (_r["extra"][2]
                             if len(_r["extra"]) > 2 else None),
                            _r["iter"])
                    _window_proposed = False
                    if windowed_stop.enabled:
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
                            _window_proposed = windowed_stop.observe(
                                _window_it, _r["loss"], _step
                            )
                        if (_window_proposed and pose_pre is not None
                                and hasattr(windowed_stop,
                                            "note_barred_proposal")):
                            _window_anomalous = (
                                float(pose_pre.anomalous()) > 0.5
                            )
                            _window_drifted = (
                                float(pose_pre.drifted()) > 0.5
                            )
                            # A start-of-frame anomaly is a frozen historical
                            # comparison.  The opt-in incumbent release can
                            # override it only after repeated complete
                            # convergence proposals.  Live step drift remains
                            # an unconditional veto.  Rules without the new
                            # interface retain the historical OR gate.
                            if hasattr(windowed_stop, "health_veto"):
                                _window_bad = windowed_stop.health_veto(
                                    _window_anomalous, _window_drifted
                                )
                            else:
                                _window_bad = (
                                    _window_anomalous or _window_drifted
                                )
                            if _window_bad:
                                windowed_stop.note_barred_proposal()
                                grad_reuse.reset_trust_lease()
                                _trust_vetoed = True
                                _window_proposed = False
                        if not _window_skip:
                            windowed_sweep.observe(
                                _window_it, _r["loss"], _step
                            )
                    if _r["loss"] < _best_loss_host:
                        _best_loss_host = _r["loss"]
                        _best_rot_host = _r["rot"].clone()
                        _best_tran_host = _r["tran"].clone()
                    _legacy_stop = stopper.check(
                        _r["iter"], _r["loss"], _r["pose_delta"]
                    )
                    if _legacy_stop:
                        _gr_stop_reason = "legacy"
                        return True
                    if _window_proposed and windowed_stop.active:
                        _gr_stop_reason = "window"
                        return True
                if _latest_trust is not None and not _trust_vetoed:
                    grad_reuse.note_batched_check(*_latest_trust)
                return False

            _window_phase_signature = None
            _stop_now = False
            # PER-FRAME ADAPTIVE-TRANSITION STATE. The switch is decided fresh
            # each frame; adaptive_switches in the preconditioner summary then
            # reports how many frames actually switched and where, which is
            # what makes a learned transition readable instead of implicit.
            _ad_enabled = (pose_pre is not None
                           and getattr(pose_pre, "adaptive_diag", False))
            _ad_active = False
            _ad_pending_loss = None
            _ad_pending_q = None
            _ad_pending_t = None
            _ad_failures = 0
            _ad_switch_iteration = None
            _ad_cur = None
            # A frame that never switches is RIGHT-CENSORED evidence, not a
            # missing sample - it says no transition was needed before the
            # frame ended. Captured at frame start because adaptive_diag flips
            # to False the moment calibration completes.
            _ad_calibration_frame = (
                _ad_enabled
                and pose_pre.adaptive_diag_calibration_frames > 0)
            # SHADOW CALIBRATION. The accepted transition runs for real while
            # the adaptive rule is merely WATCHED, so the frames that supply
            # the evidence are not the frames it damages. Observation is valid
            # only below diag_after: past that the steps are diagonal and
            # "would the full step have improved" has no answer, so such a
            # frame is recorded as censored exactly like one that converged
            # before switching.
            _sh_active = (
                pose_pre is not None
                and _diag_shadow_cfg.get("enabled", False)
                and _sh_installed is False
                and getattr(pose_pre, "diag_after", 0) > 0)
            _sh_pending_loss = None
            _sh_failures = 0
            _sh_switch_iter = None
            while True:
                iter_start_time = time.time()
                # Pass tile mask only during the sparse phase
                _is_sparse_iter = is_sparse_phase(iter, num_iters_tracking, _ps_cfg)
                if _force_dense:
                    # THE DENSE PASS THE EARLY STOP WOULD OTHERWISE SKIP.
                    _is_sparse_iter = False
                    if not _fd_seen[0]:
                        _fd_seen[0] = True
                        print(f"\n[FinalDense] first full-resolution stopping "
                              f"iteration: frame {time_idx}, iter {iter}",
                              flush=True)
                _active_tile_mask = _tile_mask if _is_sparse_iter else None
                if windowed_stop.enabled:
                    _use_pre_now = (
                        pose_pre is not None
                        and (_pre_handoff <= 0 or iter < _pre_handoff)
                    )
                    if not _use_pre_now:
                        _window_update_phase = "adam"
                    elif (getattr(pose_pre, "diag_after", 0) > 0
                          and iter >= pose_pre.diag_after):
                        _window_update_phase = "pre-diagonal"
                    elif (getattr(pose_pre, "restart_at", 0) > 0
                          and iter >= pose_pre.restart_at):
                        _window_update_phase = "pre-restarted"
                    else:
                        _window_update_phase = "pre-full"
                    _window_signature = (
                        _active_tile_mask is not None,
                        _window_update_phase,
                    )
                    if _window_phase_signature is None:
                        _window_phase_signature = _window_signature
                    elif _window_signature != _window_phase_signature:
                        # The iteration-20 diagonal transition can fall inside
                        # an eight-row device batch. Drain the old phase before
                        # resetting both statistical windows and ring order.
                        _phase_stop = _consume(es_signals.drain_phase())
                        _phase_label = (
                            f"sparse={int(_window_signature[0])},"
                            f"update={_window_signature[1]}"
                        )
                        _phase_it = (_clock.before(iter)
                                     if _clock is not None else iter)
                        windowed_stop.start_phase(_phase_it, _phase_label)
                        windowed_sweep.start_phase(_phase_it, _phase_label)
                        grad_reuse.reset_trust_lease()
                        _window_phase_signature = _window_signature
                        if _phase_stop:
                            # All completed rows from the old regime have been
                            # consumed and their best candidate retained. Stop
                            # before issuing an iteration from the new regime.
                            _stop_now = True
                            break
                # THE OBJECTIVE CHANGES HERE, ONCE PER FRAME. Within a phase
                # the mask is fixed (built once above, before this loop), so
                # consecutive gradients share a pixel subset and a secant pair
                # measures curvature. Across the flip they do not, and the pair
                # would describe the change of objective instead. Harmless for
                # an averaging metric, poisonous for a secant one.
                if (pose_pre is not None and pose_pre.bfgs
                        and _is_sparse_iter != _prev_sparse_iter):
                    pose_pre.note_objective_change()
                _prev_sparse_iter = _is_sparse_iter
                pixel_sampler.note_iter(_is_sparse_iter)
                # HANDOFF SWEEP. Launch a GN probe from whatever pose Adam has
                # reached after `iter` iterations. Read-only: it renders from
                # poses it invents and never writes to params, so tracking
                # continues exactly as if it were absent.
                #
                # WHY THE SWEEP RATHER THAN A GUESSED HANDOFF POINT. The first
                # combined-residual run showed GN succeeding from a 5.82 cm
                # start and stalling from 8.55 and 8.77 cm - a basin boundary
                # somewhere between, at n=3. Sweeping k finds it directly and
                # gives the optimal split in the same run.
                if _gn_render_fn is not None:
                    _dev = params['means3D'].device
                    _b = torch.eye(4, device=_dev)
                    _b[:3, :3] = build_rotation(
                        F.normalize(params['cam_unnorm_rots'][..., time_idx].detach()))
                    _b[:3, 3] = params['cam_trans'][..., time_idx].detach()
                    # ADAM'S TRAJECTORY, IN THE PROBE'S UNITS. Adam minimises a
                    # loss in its own units, so the two curves are not
                    # comparable unless Adam's poses are re-evaluated with the
                    # probe's cost function. Without this there is no baseline
                    # and the table cannot test the claim at all - which is
                    # what the first three runs were missing.
                    gn_probe.note_adam(time_idx, iter,
                                       gn_probe.eval_cost(_gn_render_fn, _b),
                                       _b[:3, 3].tolist())
                    if gn_probe.should_probe_at(time_idx, iter):
                        gn_probe.run(time_idx, _gn_render_fn, _b, _gn_gt,
                                     handoff=iter,
                                       render_grad_fn=_gn_render_grad_fn)
                # Gradient-variance probe. Default off; fires on a handful of
                # (frame, iteration) points and runs ~46 extra forward+backward
                # passes at each, so it is a diagnostic run, never a timed one.
                # Either probe wanting this point sets up the shared
                # isolated backward below; each then decides for itself.
                if (grad_var_probe.should_probe(time_idx, iter)
                        or tile_cost_probe.should_probe(time_idx, iter)
                        or stale_probe.wants(time_idx, iter)):
                    _gv_H = tracking_curr_data['im'].shape[1]
                    _gv_W = tracking_curr_data['im'].shape[2]
                    _gv_it = iter

                    def _gv_backward(_m):
                        # ISOLATED `variables`. get_loss updates max_2D_radius
                        # as a running max and republishes means2D for
                        # densification, so 46 probe backwards would inflate
                        # both and feed mapping a corrupted map. The probe gets
                        # a copy; the real loop's dict is never touched.
                        _v = {k: (v.clone() if torch.is_tensor(v) else v)
                              for k, v in variables.items()}
                        optimizer.zero_grad(set_to_none=False)
                        _l, _, _ = get_loss(
                            params, tracking_curr_data, _v, iter_time_idx,
                            config['tracking']['loss_weights'],
                            config['tracking']['use_sil_for_loss'],
                            config['tracking']['sil_thres'],
                            config['tracking']['use_l1'],
                            config['tracking']['ignore_outlier_depth_loss'],
                            tracking=True, plot_dir=eval_dir,
                            visualize_tracking_loss=False,
                            tracking_iteration=_gv_it, tile_mask=_m,
                            mask_multiply=_mask_multiply,
                            # None, NOT the live binning kwargs: the probe must
                            # not write the shared count/overflow buffers that
                            # note_iteration() and end_frame() read. get_loss
                            # does `binning_kwargs or {}`, so this takes the
                            # rasterizer's ordinary readback path.
                            binning_kwargs=None)
                        _l.backward()
                        return float(_l.detach())

                    def _gv_mask(_ratio, _gfrac):
                        return build_tile_mask(
                            tracking_curr_data['im'], _gv_H, _gv_W,
                            {'sample_ratio': _ratio, 'gradient_frac': _gfrac})

                    if grad_var_probe.should_probe(time_idx, iter):
                        grad_var_probe.run(
                            time_idx, iter,
                            [params['cam_unnorm_rots'], params['cam_trans']],
                            _gv_backward, _gv_mask, _ps_cfg)
                    # Shares _gv_backward deliberately. That closure already
                    # isolates `variables`, so a probe backward cannot inflate
                    # max_2D_radius or republish means2D into densification,
                    # and already passes binning_kwargs=None so it cannot write
                    # the shared count/overflow buffers. A second closure would
                    # have to re-derive both, and getting either wrong corrupts
                    # the host run rather than the probe.
                    if tile_cost_probe.should_probe(time_idx, iter):
                        tile_cost_probe.run(
                            time_idx, iter, _gv_backward,
                            tracking_curr_data['im'], _gv_H, _gv_W, _ps_cfg,
                            # MUST be passed: the real iteration below calls
                            # _loss.backward() without zeroing first, so a
                            # probe gradient left on these would be added to
                            # it. The probe saves and restores them.
                            pose_tensors=[params['cam_unnorm_rots'],
                                          params['cam_trans']])
                    if stale_probe.wants(time_idx, iter):
                        # The pose as a 6-vector in the tracker's own
                        # [rho | theta] order, so differences between two
                        # iterates are the tangent actually travelled and can
                        # be multiplied by the preconditioner's 6x6 directly.
                        def _stale_tangent(_ti=time_idx):
                            with torch.no_grad():
                                _sq = F.normalize(
                                    params['cam_unnorm_rots'][..., _ti]
                                    .detach().reshape(1, 4))
                                _sT = np.eye(4)
                                _sT[:3, :3] = (build_rotation(_sq)
                                               .reshape(3, 3).cpu().numpy())
                                _sT[:3, 3] = (params['cam_trans'][..., _ti]
                                              .detach().reshape(3).cpu().numpy())
                            return _se3_log_numpy(_sT)

                        # THE GRADIENT, MAPPED INTO THE 6-D SE(3) TANGENT
                        # with the same exact analytic Jacobian the
                        # preconditioner's step uses (validated at
                        # cos=1.0000000000 against central differences).
                        # Without this the probe compares 7-vectors, which
                        # carry the quaternion gauge direction - a component
                        # that moves no pose - and cannot be multiplied by
                        # the 6x6 M at all.
                        def _stale_grad(_ti=time_idx):
                            _gq = params['cam_unnorm_rots'].grad
                            _gt = params['cam_trans'].grad
                            if _gq is None or _gt is None:
                                return None
                            with torch.no_grad():
                                return quat_trans_grad_to_tangent(
                                    params['cam_unnorm_rots'][..., _ti].reshape(4),
                                    params['cam_trans'][..., _ti].reshape(3),
                                    _gq[..., _ti].reshape(4),
                                    _gt[..., _ti].reshape(3)).reshape(-1)

                        stale_probe.observe(
                            time_idx, iter,
                            [params['cam_unnorm_rots'], params['cam_trans']],
                            _gv_backward,
                            grad_map_fn=_stale_grad,
                            tangent_fn=_stale_tangent,
                            # M is a gradient SECOND MOMENT, not a Hessian -
                            # proportional to the Gauss-Newton one with an
                            # unknown constant, which is why the probe scans
                            # the scale and reports an upper bound.
                            curvature_fn=(lambda: None if pose_pre is None
                                          else pose_pre.M))
                # Render, loss, backward and optimizer step, as one callable so
                # TrackingIterationGraph can record the whole thing into a CUDA
                # graph. Everything that synchronises or branches on device
                # values stays outside it, below.
                _bin_kwargs = binning_capacity.render_kwargs()
                _tracking_it = iter
                # Refreshed every iteration: transform_to_frame builds a
                # NEW transformed_pts each time, so a stale handle would
                # feed the metric a gradient from the previous pose. None
                # when the arm is off - that is what makes get_loss skip
                # retain_grad and costs nothing on the default path.
                _pose_samples = ({} if (pose_pre is not None
                                        and pose_pre.sample_metric)
                                 else None)
                _trace_this_step = (excursion_recorder is not None
                                    and not excursion_recorder.written
                                    and pose_pre is not None
                                    and (_pre_handoff <= 0
                                         or iter < _pre_handoff))
                if _trace_this_step:
                    # Diagnostic-only synchronisation. These copies are made
                    # before the iteration so the trace records the exact pose
                    # that produced the loss alongside the pose after the
                    # update. Ordinary runs never execute this branch.
                    _trace_q_before = (params['cam_unnorm_rots'][..., time_idx]
                                       .detach().reshape(4).clone())
                    _trace_t_before = (params['cam_trans'][..., time_idx]
                                       .detach().reshape(3).clone())
                if _pre_handoff > 0 and pose_pre is not None and iter == _pre_handoff:
                    # FRESH ADAM STATE AT THE SWITCH. Carrying m and v from a
                    # different update rule would inject a wrong first step -
                    # the same reasoning carry_frame uses to reset m every
                    # frame while carrying M.
                    #
                    # It is already fresh by construction: `optimizer` is built
                    # per frame (see initialize_optimizer above) and its step()
                    # is never called during the preconditioner phase. This
                    # ASSERTS that rather than assuming it, because if either
                    # fact ever changes the arm would silently start Adam with
                    # momentum from the previous frame's refinement.
                    for _p in (params['cam_unnorm_rots'], params['cam_trans']):
                        _st = optimizer.state.get(_p, None)
                        if _st:
                            print(f"[Handoff] WARNING: Adam state was NOT empty "
                                  f"at the switch ({list(_st)}) - resetting",
                                  flush=True)
                            optimizer.state[_p] = {}

                # GRADIENT REUSE. The whole render, loss and backward are
                # skipped - not one stage of them - which is why this is
                # worth more than tile masking (6-18% for 75% of tiles) or
                # binning reuse (null) measured on this branch.
                #
                # Justified by utils/grad_staleness.py at the SHIPPED
                # operating point: one step stale the pose gradient still
                # reads cos 0.9473 at 0.942 of the magnitude. Two steps
                # stale it is 0.5141, and four steps it is ANTI-PARALLEL,
                # which is why GradReuse refuses a period above 2.
                #
                # A reuse iteration returns the PREVIOUS loss, because it
                # computes none. Downstream that repeat reads as 'no
                # improvement' to an incumbent stopping rule, so stopping
                # tends to fire earlier. Real interaction, watch it in
                # iterations/frame.
                #
                # DECIDED OUT HERE so the caller can bypass the iteration
                # graph on a reuse iteration. restore_grads() is a copy_
                # into the existing .grad, never a rebind, so the addresses
                # a capture recorded stay valid across it.
                _gr_candidate[0] = grad_reuse.should_reuse(
                    _tracking_it,
                    _tracking_it >= num_iters_tracking - 1,
                )
                _gr_reused[0] = (
                    _gr_apply_reuse
                    and _gr_candidate[0]
                    and restore_grads([params['cam_unnorm_rots'],
                                       params['cam_trans']], _gr_stash[0]))
                # COUNTED OUT HERE TOO, and leaving these inside would be a
                # REPORTING bug under the graph: a rendered iteration
                # REPLAYS, so its Python body never executes and
                # note_rendered() would only ever count the warmup and the
                # capture - the summary would read thousands reused against
                # a handful rendered.
                # DROP THE PREVIOUS ITERATION BEFORE THE CAPTURE RUNS, for
                # the same reason as the `loss = losses = None` below: a
                # live reference to iteration N-1 keeps its warmup-era
                # accumulators alive and the capture reuses nodes stamped
                # with the legacy default stream. Gradient reuse
                # reintroduces the hazard because holding the previous
                # iteration IS the mechanism.
                if not _gr_reused[0] and tracking_iteration_graph.will_capture():
                    _gr_last[0] = None
                    _gr_stash[0] = None
                if _clock is not None:
                    _clock.note(bool(_gr_reused[0]))
                if _gr_reused[0]:
                    grad_reuse.note_reused()
                elif grad_reuse.enabled:
                    # Gated so the counters stay at zero when reuse is off.
                    grad_reuse.note_rendered()
                    if _gr_candidate[0]:
                        # Shadow mode rendered this iteration, but consume the
                        # same bounded trust credit the applying arm would have
                        # consumed.  This keeps subsequent check decisions and
                        # their cadence comparable without falsifying the
                        # rendered/reused counters.
                        grad_reuse.note_reuse_candidate()
                def _tracking_iteration(_vars=variables, _mask=_active_tile_mask,
                                        _bk=_bin_kwargs, _it=_tracking_it,
                                        _ho=_pre_handoff):
                    nonlocal _ad_active, _ad_pending_loss, _ad_pending_q
                    nonlocal _ad_pending_t, _ad_failures, _ad_switch_iteration
                    nonlocal _ad_cur
                    nonlocal _sh_pending_loss, _sh_failures, _sh_switch_iter
                    # Snapshot the pose before the step so the pose delta can be
                    # differenced on-device at the end of this same captured
                    # region - no eager cat/norm and no .item() outside it.
                    es_signals.record_pre_step(params['cam_unnorm_rots'][..., time_idx],
                                               params['cam_trans'][..., time_idx])
                    if _gr_reused[0]:
                        # THE LIVE _vars, NOT THE STASHED ONE.
                        #
                        # get_loss returns a variables dict carrying
                        # max_2D_radius and a means2D whose .grad
                        # densification accumulates from. Handing back the
                        # PREVIOUS one makes the caller reassign `variables`
                        # to a dict holding a gradient that has already been
                        # counted, so a reuse iteration feeds densification a
                        # second helping of the last render's statistics.
                        #
                        # Measured cost of getting this wrong: PSNR 26.61 ->
                        # 23.63 while ATE went 3.13 -> 2.91. Tracking fine,
                        # map degraded - the signature of a densification
                        # problem rather than a pose one.
                        #
                        # A reuse iteration renders nothing, so it has no new
                        # statistics to contribute and `variables` should pass
                        # through untouched.
                        _loss, _losses = _gr_last[0][0], _gr_last[0][2]
                        _vars_out = _vars
                    else:
                        _loss, _vars_out, _losses = get_loss(
                            params, tracking_curr_data, _vars, iter_time_idx,
                            config['tracking']['loss_weights'],
                            config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                            config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'],
                            tracking=True,
                            plot_dir=eval_dir,
                            visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                            tracking_iteration=_it, tile_mask=_mask,
                            mask_multiply=_mask_multiply,
                            binning_kwargs=_bk,
                            pose_samples=_pose_samples)
                        _loss.backward()
                        if (grad_reuse.enabled and not _gr_apply_reuse
                                and pose_pre is None):
                            # The accepted path obtains this tangent inside
                            # the preconditioner update.  The Adam baseline has
                            # no such update, but its reuse-off control still
                            # needs to pay for and execute the same trust check.
                            # Graphs are off in that stage, so this is ordinary
                            # eager, fixed-shape device arithmetic.
                            with torch.no_grad():
                                _gr_gxi_out[0] = quat_trans_grad_to_tangent(
                                    params['cam_unnorm_rots'][..., time_idx].reshape(4),
                                    params['cam_trans'][..., time_idx].reshape(3),
                                    params['cam_unnorm_rots'].grad[..., time_idx].reshape(4),
                                    params['cam_trans'].grad[..., time_idx].reshape(3))
                        # Before the step: the Adam path clears .grad in it.
                        iter_trace.capture(
                            params['cam_unnorm_rots'][..., time_idx],
                            params['cam_trans'][..., time_idx],
                            params['cam_unnorm_rots'].grad[..., time_idx],
                            params['cam_trans'].grad[..., time_idx])
                        # BETWEEN backward() and the step, never after: the
                        # preconditioner clears .grad with .zero_() so the
                        # addresses stay stable for a capture, and after the
                        # step there is nothing left to stash.
                        if grad_reuse.enabled:
                            _gr_stash[0] = stash_grads(
                                [params['cam_unnorm_rots'], params['cam_trans']])
                            _gr_last[0] = (_loss, _vars_out, _losses)
                    # PHASE SPLIT. Before the handoff the preconditioner steps;
                    # after it, plain Adam does. `_it` is the within-frame
                    # iteration index, so the split is per frame, not per run.
                    _use_pre = (pose_pre is not None
                                and (_ho <= 0 or _it < _ho))
                    if _use_pre and pose_pre.tangent:
                        # STAGE 1. The render path is untouched - autograd fills
                        # grads on (q, t) exactly as before. Only the STEP moves
                        # to the tangent: map the 7-vector gradient across with
                        # the exact analytic Jacobian of the parameterisation
                        # (validated against central differences at
                        # cos=1.0000000000, rel_err 3.5e-11, with the textbook
                        # V under an eps sweep - unlike the RENDERER's backward,
                        # which was flat at cos=0.868 and killed the GN line),
                        # precondition in 6-D where M is full rank and there is
                        # no null direction, then apply T <- exp(dxi) T.
                        # Wrapped as a closure so TrackingStepGraph can CAPTURE
                        # it. Everything inside is fixed-shape, branch-free and
                        # sync-free - se3_exp_capturable / mat_to_quat_capturable
                        # replace the gn_tracking versions, which branch on a
                        # host-side float that a capture would freeze. Grads are
                        # zeroed with .zero_() rather than set_to_none so their
                        # addresses stay stable across replays.
                        def _precond_update():
                            with torch.no_grad():
                                _q = params['cam_unnorm_rots'][..., time_idx].reshape(4)
                                _t = params['cam_trans'][..., time_idx].reshape(3)
                                _gxi = quat_trans_grad_to_tangent(
                                    _q, _t,
                                    params['cam_unnorm_rots'].grad[..., time_idx].reshape(4),
                                    params['cam_trans'].grad[..., time_idx].reshape(3))
                                # HAND THE FRESH TANGENT OUT OF THE CAPTURE.
                                # Calling note_fresh_grad here would be wrong:
                                # Python executes once at capture and never on
                                # replay, and its batched .item() sync cannot
                                # live inside a CUDA graph.  The tensor itself
                                # is replay-stable, so the eager block after
                                # tracking_iteration_graph.run() consumes it.
                                _gr_gxi_out[0] = _gxi
                                # THE SAMPLE METRIC. `_gxi` above is a SUM
                                # over Gaussians of contributions the backward
                                # already computed and then threw away. In
                                # isotropic mode the pose reaches the loss only
                                # through transformed_pts, so its retained grad
                                # recovers every one of them, and
                                #     G.sum(0) == _gxi
                                # exactly - the invariant group 14 of
                                # utils/test_pose_preconditioner.py asserts to
                                # 1e-14, which is why this needed no GPU time to
                                # validate. G^T G is one 6xNx6 gemm against an
                                # ~11 ms render, and it is FULL RANK from the
                                # first backward of the first frame, where the
                                # rank-1 EMA needs at least six iterations.
                                _Ms = None
                                if _pose_samples:
                                    _pts = _pose_samples['pts']
                                    # A dropped grad must not silently fall back
                                    # to the rank-1 path - that is an arm that
                                    # reports as the new method and measures the
                                    # old one. pose_pre.sampled vs steps in the
                                    # summary is the check; this keeps it honest.
                                    if _pts.grad is not None:
                                        _G = per_gaussian_tangent_grads(
                                            _pts.detach(), _pts.grad)
                                        _Ms = sample_metric(_G,
                                                            trace_match=True)
                                        _pre_neff.add_(effective_samples(_G))
                                        _pre_cos.add_(rank1_alignment(_G))
                                        _pre_neff_n.add_(1)
                                # A reused gradient is the same measurement
                                # twice, so under GRAD_REUSE_NOACC it drives a
                                # step without updating m or M.
                                _dxi = pose_pre.step(
                                    _gxi, M_inst=_Ms,
                                    accumulate=not (_gr_reused[0]
                                                    and grad_reuse.no_accum),
                                    accumulate_m=not (_gr_reused[0]
                                                      and (grad_reuse.no_accum
                                                           or grad_reuse.freeze_m)))
                                if precond_probe.active:
                                    # AFTER step(), so M_real is the metric
                                    # this iteration's step was actually
                                    # taken through, not the one before it.
                                    # M_inst is only available in sample mode;
                                    # the rel_grad / rank / conditioning columns
                                    # do not need it, and those are what a
                                    # STOPPING question asks for. Requiring it
                                    # made the probe silently inert on every
                                    # arm that actually ships.
                                    precond_probe.record(
                                        _gxi,
                                        _Ms if _Ms is not None else pose_pre.M,
                                        pose_pre.M, pose_pre.rel_grad())
                                _qn, _tn = apply_tangent_step(
                                    _q, _t, _dxi, mat_to_quat_capturable)
                                # exp() preserves the norm exactly, so this
                                # stays 1 by construction rather than by
                                # renormalisation. Kept purely to prove that.
                                pose_pre.note_gauge(_qn.norm())
                                params['cam_unnorm_rots'][..., time_idx] = _qn.to(
                                    params['cam_unnorm_rots'].dtype).reshape(
                                        params['cam_unnorm_rots'][..., time_idx].shape)
                                params['cam_trans'][..., time_idx] = _tn.to(
                                    params['cam_trans'].dtype).reshape(
                                        params['cam_trans'][..., time_idx].shape)
                                for _p in (params['cam_unnorm_rots'],
                                           params['cam_trans']):
                                    if _p.grad is not None:
                                        _p.grad.zero_()
                        # THE ADAPTIVE TRANSITION DECIDES HERE, between
                        # backward() and the step, because it scores the
                        # PREVIOUS full-matrix proposal using this render's
                        # loss and must be able to undo it before this
                        # iteration commits anything on top.
                        #
                        # A full step taken at iteration k from pose P_k with
                        # loss L_k is a proposal. Iteration k+1 renders at the
                        # resulting pose giving L_{k+1}. If that does not
                        # improve on L_k it is one failure; after `patience`
                        # consecutive failures the pose returns to P_k and the
                        # diagonal step - from the same retained gradient and
                        # metric - replaces the proposal.
                        # SHADOW OBSERVATION. Identical arithmetic to the
                        # applying path, and deliberately no action: nothing is
                        # rolled back, nothing is latched, the frame proceeds
                        # exactly as the accepted arm does.
                        if (_sh_active and _sh_switch_iter is None
                                and _it < pose_pre.diag_after):
                            _sh_cur = float(_loss.detach())
                            if _sh_pending_loss is not None:
                                _sh_failures, _sh_hit = adaptive_failure_streak(
                                    _sh_failures,
                                    adaptive_loss_improved(_sh_pending_loss,
                                                           _sh_cur),
                                    _diag_shadow_cfg.get("patience", 1))
                                if _sh_hit:
                                    _sh_switch_iter = _it + 1
                            _sh_pending_loss = _sh_cur

                        _ad_reject = False
                        if _ad_enabled and not _ad_active:
                            # The one sync this arm adds. It stops when the
                            # frame latches, and stops for the whole run once
                            # calibration installs a fixed diag_after.
                            _ad_cur = float(_loss.detach())
                            if _ad_pending_loss is not None:
                                _ad_failures, _ad_reject = adaptive_failure_streak(
                                    _ad_failures,
                                    adaptive_loss_improved(_ad_pending_loss,
                                                           _ad_cur),
                                    pose_pre.adaptive_diag_patience)
                                # activate_adaptive_diagonal raises without a
                                # metric; hold the rejection rather than crash.
                                if _ad_reject and not pose_pre._have_metric:
                                    _ad_reject = False
                        if _ad_reject:
                            with torch.no_grad():
                                _dxi = pose_pre.activate_adaptive_diagonal()
                                _qn, _tn = apply_tangent_step(
                                    _ad_pending_q, _ad_pending_t, _dxi,
                                    mat_to_quat_capturable)
                                pose_pre.note_gauge(_qn.norm())
                                params['cam_unnorm_rots'][..., time_idx] = _qn.to(
                                    params['cam_unnorm_rots'].dtype).reshape(
                                        params['cam_unnorm_rots'][..., time_idx].shape)
                                params['cam_trans'][..., time_idx] = _tn.to(
                                    params['cam_trans'].dtype).reshape(
                                        params['cam_trans'][..., time_idx].shape)
                                for _p in (params['cam_unnorm_rots'],
                                           params['cam_trans']):
                                    if _p.grad is not None:
                                        _p.grad.zero_()
                            _ad_active = True
                            _ad_switch_iteration = _it + 1
                            _ad_pending_loss = None
                            _ad_pending_q = _ad_pending_t = None
                            _ad_failures = 0
                        else:
                            if _ad_enabled and not _ad_active:
                                # Score THIS step at the next render.
                                _ad_pending_loss = _ad_cur
                                _ad_pending_q = params['cam_unnorm_rots'][
                                    ..., time_idx].detach().clone().reshape(4)
                                _ad_pending_t = params['cam_trans'][
                                    ..., time_idx].detach().clone().reshape(3)
                            tracking_step_graph.step(fn=_precond_update)
                    elif _use_pre:
                        # Order is fixed for the whole run: the concatenation
                        # defines what M's rows and columns mean.
                        # The gauge direction in the 7-vector: scaling the
                        # quaternion changes nothing the loss can see, so the
                        # direction is (q/|q|, 0, 0, 0) in the same
                        # concatenation order as the pairs below. Projecting it
                        # out stops the preconditioner spending its trust
                        # region on a direction with no curvature - which it is
                        # otherwise drawn to, since low curvature earns a long
                        # step. See PosePreconditioner.step_params.
                        with torch.no_grad():
                            _qv = params['cam_unnorm_rots'][..., time_idx].reshape(-1)
                            _gauge = torch.cat([
                                _qv / _qv.norm().clamp_min(1e-20),
                                torch.zeros(3, dtype=_qv.dtype, device=_qv.device),
                            ])
                        pose_pre.step_params([
                            (params['cam_unnorm_rots'], time_idx),
                            (params['cam_trans'], time_idx),
                        ], gauge=_gauge)
                        # GAUGE FIX. cam_unnorm_rots is F.normalize'd at use, so
                        # scaling it is an exact null direction of the loss -
                        # zero curvature, which is exactly the direction the
                        # Levenberg floor hands the LONGEST step to. Adam is
                        # immune (it divides by sqrt(v), so no signal means no
                        # step); a preconditioner is actively attracted to it,
                        # and the drifting norm silently rescales the effective
                        # rotation step size. Measured before this fix: matched
                        # the Adam control frame-for-frame to ~frame 14, then
                        # diverged.
                        #
                        # Renormalising here is a no-op for the loss, which
                        # cannot see the norm, and removes the drift entirely.
                        with torch.no_grad():
                            _q = params['cam_unnorm_rots'][..., time_idx]
                            _qn = _q.norm()
                            pose_pre.note_gauge(_qn)
                            _q.div_(_qn.clamp_min(1e-8))
                        for _p in (params['cam_unnorm_rots'], params['cam_trans']):
                            if _p.grad is not None:
                                _p.grad.zero_()
                    else:
                        if (pose_pre is not None and pose_pre.tangent
                                and _ho > 0):
                            # ADAM PHASE OF A HANDOFF. Keep feeding the plateau
                            # tracker: step() is not being called, so _g_best
                            # and the stall counter would FREEZE at the handoff
                            # iteration and a frame that had not already stalled
                            # could never stop. Observed as roughly half the
                            # frames at exactly 200/200.
                            #
                            # Mapped to the SE(3) tangent first, because
                            # _g_best must compare like with like - a
                            # 7-parameter (q, t) norm is a different quantity.
                            with torch.no_grad():
                                pose_pre.observe(quat_trans_grad_to_tangent(
                                    params['cam_unnorm_rots'][..., time_idx].reshape(4),
                                    params['cam_trans'][..., time_idx].reshape(3),
                                    params['cam_unnorm_rots'].grad[..., time_idx].reshape(4),
                                    params['cam_trans'].grad[..., time_idx].reshape(3)))
                                # C0. THE MEASUREMENT THIS PHASE NEVER HAD.
                                # step() records the tangent step it requests,
                                # but step() does not run here - so the Adam
                                # phase's step SIZE has never been measured,
                                # and it is the number that decides whether the
                                # handoff supplies DIRECTION or DISTANCE.
                                # Snapshot before, difference after.
                                _c0_q.copy_(params['cam_unnorm_rots'][..., time_idx].reshape(4))
                                _c0_t.copy_(params['cam_trans'][..., time_idx].reshape(3))
                        elif adam_probe is not None:
                            # Same before/after pattern as the handoff branch
                            # above, into the persistent _c0_* buffers - a
                            # .clone() here would allocate from the graph's
                            # private pool on a captured replay.
                            with torch.no_grad():
                                _c0_q.copy_(params['cam_unnorm_rots'][..., time_idx].reshape(4))
                                _c0_t.copy_(params['cam_trans'][..., time_idx].reshape(3))
                        tracking_step_graph.step(optimizer)
                        if adam_probe is not None and not (
                                pose_pre is not None and pose_pre.tangent
                                and _ho > 0):
                            with torch.no_grad():
                                adam_probe.record(tangent_of_pose_delta(
                                    _c0_q, _c0_t,
                                    params['cam_unnorm_rots'][..., time_idx].reshape(4),
                                    params['cam_trans'][..., time_idx].reshape(3)), _it)
                        if (pose_pre is not None and pose_pre.tangent
                                and _ho > 0):
                            # AFTER the step, against the snapshot above. The
                            # SE(3) log puts Adam's realised motion in the same
                            # units as _step_sum; a 7-parameter (q, t) delta
                            # norm is a different quantity and comparing it
                            # would answer a different question. Normalisation
                            # inside tangent_of_pose_delta drops the quaternion
                            # gauge, which Adam moves freely and which the loss
                            # cannot see.
                            with torch.no_grad():
                                pose_pre.note_applied_step(tangent_of_pose_delta(
                                    _c0_q, _c0_t,
                                    params['cam_unnorm_rots'][..., time_idx].reshape(4),
                                    params['cam_trans'][..., time_idx].reshape(3)))
                    # Loss, pose delta and the stepped pose, straight into the
                    # ring buffer. Everything the host needs for early stopping
                    # AND candidate selection now leaves the iteration through
                    # one drain per `batch` iterations instead of three syncs
                    # per iteration.
                    es_signals.record_post_step(_loss,
                                                params['cam_unnorm_rots'][..., time_idx],
                                                params['cam_trans'][..., time_idx])
                    return _loss, _vars_out, _losses

                # Backprop
                if not es_signals.enabled:
                    # With batching on, the pose snapshot happens inside the
                    # captured region instead (record_pre_step), so these two
                    # eager clones are pure waste.
                    prev_rot   = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                    prev_trans = params['cam_trans'][..., time_idx].detach().clone()
                # Release the previous iteration's autograd graph before the
                # capture iteration runs. These names still hold iteration N-1's
                # tensors while iteration N captures, and that is what blocked
                # CUDA graph capture.
                #
                # Autograd stamps every Node with the stream current at forward
                # time, and AccumulateGrad is cached per-leaf via a weak_ptr. A
                # live reference to a previous loss keeps the warmup-era
                # accumulators alive, so the capture reuses nodes stamped with
                # the legacy default stream instead of rebuilding them on the
                # capture stream. Drop the references and the weak_ptrs expire,
                # so they are rebuilt inside the capture.
                #
                # Isolated in profiling/repro_capture.py: rungs 4a-4c pass and
                # 4d fails on the single addition of a live reference to the
                # previous loss; the same test with the reference dropped (fixB)
                # passes. Note 4b passing means warmup on the default stream is
                # by itself harmless - the pin is the necessary ingredient.
                loss = losses = None
                loss, variables, losses = tracking_iteration_graph.run(
                    _tracking_iteration, bypass=_gr_reused[0])
                # ADAPTIVE REUSE TRUST SIGNAL, OUTSIDE CUDA CAPTURE.  Restored
                # gradients on reuse iterations are exact copies of the last
                # fresh one, so feeding them would manufacture cos=1.  None is
                # safe (for an Adam-only arm or a missing gradient) and counts
                # as no evidence rather than distrust.
                iter_trace.note_iteration(
                    iter, loss, params['cam_unnorm_rots'][..., time_idx],
                    params['cam_trans'][..., time_idx], reused=_gr_reused[0])
                if (not _gr_reused[0]
                        and not (_gr_candidate[0] and not _gr_apply_reuse)):
                    _gr_trust_signal = grad_reuse.note_fresh_grad(
                        _gr_gxi_out[0], _tracking_it)
                    if _gr_trust_signal is not None:
                        es_signals.write_current_extra(_gr_trust_signal)
                # OUTSIDE the captured region on purpose. eigh is not
                # capturable (cuSOLVER allocates and may sync), so M is
                # accumulated in the step - pure arithmetic, capturable - and
                # only re-factorised here, every refactor_every iterations.
                # Between rebuilds the applied preconditioner is a constant
                # matrix, which is what a capture needs.
                if pose_pre is not None and (_pre_handoff <= 0
                                             or iter < _pre_handoff):
                    # SKIP DURING THE ADAM PHASE, for two reasons.
                    #
                    # 1. refactor() is where _step_n is counted, so counting it
                    #    on iterations that never called step() DILUTES every
                    #    step statistic by the ratio of the two. At handoff=20
                    #    with a 200 cap that is 10x: the summary read
                    #    'step |d| mean 2.15e-04 = 0.02x Adam' where the true
                    #    mean was ~2.1e-3 = 0.21x, and the arm looked like it
                    #    had collapsed when it had not.
                    # 2. It rebuilds P with a 6x6 eigh every refactor_every
                    #    iterations, for a preconditioner nothing is using.
                    pose_pre.refactor()
                if _trace_this_step:
                    _seen = variables.get('seen')
                    excursion_recorder.record({
                        "frame": int(time_idx),
                        "iteration": int(iter),
                        "loss": float(loss.detach()),
                        "seen": (int(_seen.count_nonzero().item())
                                 if torch.is_tensor(_seen) else None),
                        "pose_before": {
                            "quaternion": _trace_q_before.detach().cpu().tolist(),
                            "translation": _trace_t_before.detach().cpu().tolist(),
                        },
                        "pose_after": {
                            "quaternion": (params['cam_unnorm_rots'][..., time_idx]
                                           .detach().reshape(4).cpu().tolist()),
                            "translation": (params['cam_trans'][..., time_idx]
                                            .detach().reshape(3).cpu().tolist()),
                        },
                        "preconditioner": pose_pre.diagnostic_snapshot(),
                    })
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                _stop_now = False
                if es_signals.enabled:
                    # Nothing is read back here. loss stays untouched on the
                    # device, and the drain below syncs once every `batch`
                    # iterations instead of three times every iteration.
                    es_signals.note_iteration()
                    progress_bar.update(1)
                    _stop_now = _consume(es_signals.drain_if_full())
                else:
                    with torch.no_grad():
                        pose_delta_norm = torch.cat([
                            (params['cam_unnorm_rots'][..., time_idx].detach() - prev_rot).flatten(),
                            (params['cam_trans'][..., time_idx].detach() - prev_trans).flatten(),
                        ]).norm().item()
                        # Save the best candidate rotation & translation
                        if _sync_free_candidates:
                            # `if loss < current_min_loss` converts a CUDA tensor
                            # to a Python bool, which blocks the host until the
                            # GPU drains - one of three such syncs per tracking
                            # iteration, ~207k per run. torch.where does the same
                            # selection entirely on-device: identical values, no
                            # host round-trip. loss is detached because the
                            # running minimum is only ever compared, never
                            # backpropagated (the eager path stores the live
                            # tensor and so pins its autograd graph until the
                            # next improvement).
                            #
                            # BROKE TRACKING (ATE 64.28cm), root cause never
                            # found. early_stop.batched supersedes it: the same
                            # sync is removed, but by batching the HOST
                            # comparison rather than moving it on-device.
                            _improved = loss.detach() < current_min_loss_t
                            current_min_loss_t = torch.where(_improved, loss.detach(), current_min_loss_t)
                            # Same COMMIT_AT_LOSS substitution as the eager
                            # branch below - without it the flag would silently
                            # do nothing on this path.
                            _cand_rot = (prev_rot if _commit_at_loss
                                         else params['cam_unnorm_rots'][..., time_idx].detach())
                            _cand_tran = (prev_trans if _commit_at_loss
                                          else params['cam_trans'][..., time_idx].detach())
                            candidate_cam_unnorm_rot = torch.where(
                                _improved, _cand_rot, candidate_cam_unnorm_rot)
                            candidate_cam_tran = torch.where(
                                _improved, _cand_tran, candidate_cam_tran)
                        elif loss < current_min_loss:
                            # MUST be .clone(), not just .detach().
                            #
                            # detach() shares storage. Under the iteration graph
                            # that is fatal: replay rewrites the captured outputs
                            # in place, so `loss` is the SAME tensor every
                            # iteration. A detached alias of it therefore tracks
                            # the current loss instead of remembering the best
                            # one, and `loss < current_min_loss` compares the
                            # buffer with itself - always False. The candidate
                            # pose froze a few iterations after capture while
                            # tracking ran on to ~98, so every frame committed a
                            # barely-converged pose: 591/591 frames captured and
                            # ATE 156.40cm.
                            #
                            # The in-place rewrite is the graph's whole point -
                            # it is how loss and losses reach the early-stop
                            # check without re-running Python - so anything that
                            # has to OUTLIVE an iteration must copy out of it.
                            #
                            # clone() also keeps what detach() was for: the live
                            # tensor is not stored, so the best iteration's
                            # autograd graph is not pinned across the capture
                            # boundary. The comparison above already converts to
                            # a Python bool, so no sync is added.
                            current_min_loss = loss.detach().clone()
                            # COMMIT_AT_LOSS: pair the loss with the pose it was
                            # actually EVALUATED at.
                            #
                            # `loss` comes from get_loss at the TOP of this
                            # iteration, before backward and before the step -
                            # so it describes prev_rot/prev_trans. Reading
                            # params[...] here reads the pose the step just
                            # moved to. Every frame therefore commits a pose
                            # ONE OPTIMISER STEP past the one that achieved the
                            # best loss, and the size of that error is the size
                            # of the last step.
                            #
                            # WHICH IS WHY IT SHOWS UP AS A PSNR GAP RATHER THAN
                            # AN ATE ONE. Measured on TUM fr1, full length: the
                            # handoff's Adam tail steps at 0.49x ref where the
                            # handoff-free tail sits at 0.04-0.08x, so the
                            # handoff commits ~6-8x further from its own best
                            # pose - ~2.5 px of render misalignment against
                            # ~0.3 px. ATE 3.86 vs 3.38 (aligned, so a per-frame
                            # offset partly cancels), PSNR 19.00 vs 21.53.
                            #
                            # Present in the eager path AND in es_signals'
                            # batched path, in all three models and in the
                            # baseline. Default OFF so every recorded number
                            # stays comparable - same precedent as PRE_MAX_STEP
                            # and STOP_ANCHOR: add the flag, measure, then flip.
                            if _commit_at_loss:
                                candidate_cam_unnorm_rot = prev_rot.clone()
                                candidate_cam_tran = prev_trans.clone()
                            else:
                                candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                                candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                        # Report Progress
                        if config['report_iter_progress']:
                            if config['use_wandb']:
                                report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                                wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'])
                            else:
                                report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                        else:
                            progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                binning_capacity.note_iteration()
                # GRADIENT-NORM STOPPING, the criterion that matches a
                # preconditioned tracker. Both existing criteria were designed
                # around Adam and neither works here:
                #
                #   pose_eps  - can NEVER fire. M ~ E[gg^T] scales as g^2 and
                #               m as g, so M^{-1/2} m is exactly scale-free and
                #               the step stays ~lr at the optimum. Measured:
                #               pose_eps=1e-4 against a ~1e-3 step left every
                #               frame at the 200 cap.
                #   loss_eps  - has no scale that transfers. 0.004 stopped the
                #               hard frame 325 after 28 iterations; 1e-4 never
                #               stopped anything.
                #
                # |g| DOES decay, and |g|/|g_0| against the frame's own first
                # gradient is scale-free ACROSS frames, so a single threshold
                # holds for easy and hard frames alike - no per-scene tuning,
                # which is the property the whole method is arguing for.
                if _force_dense:
                    # The full-resolution iteration just ran. Commit and stop -
                    # unconditionally, so a criterion that no longer fires
                    # cannot leave the frame running at full resolution.
                    break
                if _pre_stop_rel > 0.0 and pose_pre is not None:
                    # STOP_REF=running judges against a running average of
                    # per-frame INITIAL gradients rather than this frame's own.
                    # The per-frame reference is trivially satisfied by a
                    # diverged frame - huge |g_0| means a 20x drop is nearly
                    # free - so it quits at the floor while still wrong, commits
                    # the pose, and the next frame starts worse: divergence
                    # makes the criterion fire sooner. Observed as every frame
                    # stopping at exactly STOP_MIN while mapping crawled at
                    # 2 it/s.
                    #
                    # CHECKED EVERY _pre_stop_every ITERATIONS, not every one.
                    # float() on a device tensor is a host sync in the hot
                    # loop; at ~67 iterations a frame that is 67 syncs to save
                    # at most a handful of iterations. Checking every k costs
                    # up to k-1 extra iterations per frame and removes
                    # (k-1)/k of the syncs.
                    if (iter >= _pre_stop_min
                            and iter % _pre_stop_every == 0):
                        # STOP_MODE=best: PATIENCE ON THE BEST GRADIENT, not a
                        # threshold on the current one. _stall counts iterations
                        # since |g| last set a new minimum for this frame, so it
                        # is monotone by construction - an overshoot spike
                        # cannot reset it and a noise dip cannot trigger it.
                        #
                        # It also reads the SAME quantity in both phases of a
                        # handoff, which loss_eps cannot: that criterion is
                        # really about step size, and Adam's steps are ~4x the
                        # preconditioner's, so no single value fits both.
                        # NO `continue` HERE, AND THAT IS NOT A STYLE POINT.
                        # An early version returned to the top of the loop from
                        # inside this branch, which skips the
                        # `if iter == num_iters_tracking` block below - the ONLY
                        # thing that terminates a frame at the cap. With
                        # stop_every=5 and a 200 cap, 200 % 5 == 0, so a frame
                        # that never stalls checked at exactly the cap, did not
                        # stop, and looped forever: observed as
                        # `Tracking Time Step: 2: 2719it`. It survived one run
                        # only because every frame happened to stall first.
                        # Both modes now compute a flag and fall through.
                        # TWO GUARDS, DELIBERATELY DIFFERENT MEASUREMENTS.
                        # anomalous() asks whether the frame STARTED badly
                        # (|g_0| vs a decaying average of |g_0|); drifted()
                        # asks whether it is STEPPING badly (|d|/ref vs a
                        # frozen average of |d|/ref). The first is blind to a
                        # slow degradation because its reference drifts with
                        # it - measured at 9-11% barred while the step
                        # distribution quadrupled over 50 frames and the run
                        # then left the frustum. Either one bars the frame from
                        # stopping early, so it runs its full budget.
                        _bad_start = (_pre_stop_ref == "running"
                                      and float(pose_pre.anomalous()) > 0.5)
                        _bad_start = _bad_start or float(pose_pre.drifted()) > 0.5
                        if _pre_stop_mode == "best":
                            _fired = (not _bad_start
                                      and float(pose_pre.stalled())
                                      >= _pre_stop_patience)
                        else:
                            # A frame that STARTED anomalously far from
                            # convergence runs its full budget, whatever the
                            # ratio says (_bad_start, computed above). This is
                            # the hard guarantee: the relative criterion is
                            # easiest to satisfy on exactly the frames that must
                            # not quit, and a gradually degrading run drags any
                            # adaptive reference along with it.
                            _r = (pose_pre.rel_grad_ref()
                                  if _pre_stop_ref == "running"
                                  else pose_pre.rel_grad())
                            _fired = (not _bad_start
                                      and float(_r) < _pre_stop_rel)
                        if _fired:
                            _gr_stop_reason = "preconditioner"
                            if _final_dense:
                                _force_dense = True
                            else:
                                break
                _es_fired = (_stop_now if es_signals.enabled
                             else stopper.check(iter - 1, loss.item(),
                                                pose_delta_norm))
                if _es_fired:
                    if not es_signals.enabled:
                        _gr_stop_reason = "legacy"
                    if _final_dense:
                        _force_dense = True
                    else:
                        break
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            if _sh_active:
                # A frame that never tripped the rule below diag_after is
                # RIGHT-CENSORED at the safe value, not a missing sample.
                _sh_samples.append(_sh_switch_iter
                                   if _sh_switch_iter is not None
                                   else int(pose_pre.diag_after))
                if _sh_switch_iter is None:
                    _sh_censored += 1
                if len(_sh_samples) >= _diag_shadow_cfg["calibration_frames"]:
                    import statistics as _stats
                    _sh_median = max(1, int(_stats.median(_sh_samples)))
                    _sh_learned = max(_sh_median,
                                      int(_diag_shadow_cfg.get("min_iter", 0)))
                    _sh_prev = int(pose_pre.diag_after)
                    # Installed exactly as the applying path does it, but only
                    # now - the frames that produced the evidence ran the
                    # accepted transition throughout.
                    pose_pre.diag_after = _sh_learned
                    pose_pre.restart_at = _sh_learned
                    _sh_installed = True
                    _q = sorted(_sh_samples)
                    _p = lambda f: _q[min(len(_q) - 1,
                                          int(f * (len(_q) - 1) + 0.5))]
                    print(
                        f"\n[Preconditioner] SHADOW calibration complete over "
                        f"{len(_sh_samples)} frames: would-switch p10/med/p90 "
                        f"= {_p(0.10)}/{_p(0.50)}/{_p(0.90)}, censored="
                        f"{_sh_censored}. Installed diag_after={_sh_learned}"
                        + (f" (median {_sh_median} floored)"
                           if _sh_learned != _sh_median else "")
                        + f", replacing the {_sh_prev} that ran during "
                        f"measurement.", flush=True)

            if _ad_calibration_frame:
                # ONE SAMPLE PER CALIBRATION FRAME: the iteration it switched
                # at, or the completed count as right-censored evidence. On the
                # Nth sample the class installs the median as diag_after and
                # clears adaptive_diag, so every later frame is an ordinary
                # fixed-D arm with a value nobody chose.
                _learned = pose_pre.finish_adaptive_calibration_frame(
                    _ad_switch_iteration, max(1, iter))
                if _learned is not None:
                    print(f"\n[Preconditioner] adaptive calibration complete: "
                          f"fixed restart_at=diag_after={_learned} learned "
                          f"from {pose_pre.adaptive_diag_calibration_frames} "
                          f"frames; later frames drop the loss readback.",
                          flush=True)

            # Closes "splatam_tracking". After every break path in the loop
            # above, so it is popped exactly once per frame.
            torch.cuda.nvtx.range_pop()
            progress_bar.close()
            binning_capacity.end_frame()
            if es_signals.enabled:
                # Flush whatever the ring still holds. Only meaningful when the
                # loop ended on the iteration budget rather than on a fire: if
                # it fired, the pending rows are the overshoot iterations that
                # the eager path would never have run, and _consume already
                # dropped them.
                if not _stop_now:
                    _consume(es_signals.drain_remainder())
                if _best_rot_host is not None:
                    candidate_cam_unnorm_rot = _best_rot_host.to(
                        candidate_cam_unnorm_rot.device,
                        dtype=candidate_cam_unnorm_rot.dtype).reshape(
                            candidate_cam_unnorm_rot.shape)
                    candidate_cam_tran = _best_tran_host.to(
                        candidate_cam_tran.device,
                        dtype=candidate_cam_tran.dtype).reshape(
                            candidate_cam_tran.shape)
            # One observation per TRACKED frame, after the final partial signal
            # batch has been drained so a real stop in that batch is not
            # mislabeled as cap.  Calibration itself no-ops after its run-in.
            grad_reuse.note_frame_stop(max(1, iter), _gr_stop_reason)
            if windowed_stop.enabled:
                windowed_stop.end_frame()
                windowed_sweep.end_frame()
            if windowed_stop.frames > _tracking_report_warmup_frames:
                _tracking_post_warmup_steps += max(1, int(iter))
                _tracking_post_warmup_frames += 1
            # COMMIT_AT_LOSS=2: SCORE THE FINAL POSE TOO.
            #
            # WHY LEVEL 1 ALONE IS NOT THE FIX. The original code committed
            # pose_{k+1} scored by loss(pose_k), which is wrong when the loss is
            # non-monotone - the step from pose_k overshot and pose_{k+1} is
            # worse. Level 1 corrects that pairing. But it also silently drops
            # the LAST pose from consideration: when a frame converges
            # monotonically the argmin is the last measured loss, the original
            # committed pose_n, and level 1 commits pose_{n-1} - throwing away a
            # real optimiser step.
            #
            # SO THE TWO LEVELS ARE OPPOSITE BIASES, and the measurements show
            # both. Level 1 on the handoff arm (Adam tail 0.49x ref, overshoots)
            # gained 2.08 dB PSNR and 0.55 cm ATE; on the restart arm (0.19x,
            # nearly monotone) it gained 0.01 dB and COST 1.2-1.5 cm of ATE.
            # The restart arm's metric split is the signature: every map metric
            # its best of any run, only ATE bad - a systematic one-step LAG
            # biases the trajectory against ground truth while leaving the map
            # self-consistent.
            #
            # Level 2 removes both biases: every candidate is scored at the pose
            # it was evaluated at, AND the final pose gets a score. One extra
            # forward per frame, no backward - about 1 render against ~29.
            if _commit_at_loss_level >= 2:
                # NOT under torch.no_grad(). get_loss(tracking=True) calls
                # rendervar['means2D'].retain_grad(), which RAISES on a tensor
                # with requires_grad=False - and no_grad makes it so. The graph
                # this builds is never backwarded and is dropped immediately
                # below, so the only cost is one forward's activations.
                # ISOLATED `variables`, for the reason grad_var_probe
                # documents: get_loss updates max_2D_radius as a running max
                # and republishes means2D for densification, so scoring a
                # pose through the live dict would feed mapping a corrupted
                # map.
                _fv = {k: (v.clone() if torch.is_tensor(v) else v)
                       for k, v in variables.items()}
                # SAME MASK AS THE LAST ITERATION USED. Losses are only
                # comparable to each other under the same pixel subset, and
                # current_min_loss holds whatever the loop produced.
                # binning_kwargs is None on purpose - it writes shared
                # count/overflow buffers that binning_capacity.end_frame()
                # has already read.
                _floss, _, _ = get_loss(
                    params, tracking_curr_data, _fv, iter_time_idx,
                    config['tracking']['loss_weights'],
                    config['tracking']['use_sil_for_loss'],
                    config['tracking']['sil_thres'],
                    config['tracking']['use_l1'],
                    config['tracking']['ignore_outlier_depth_loss'],
                    tracking=True, plot_dir=eval_dir,
                    visualize_tracking_loss=False,
                    tracking_iteration=num_iters_tracking,
                    tile_mask=_active_tile_mask,
                    mask_multiply=_mask_multiply,
                    binning_kwargs=None)
                # THE BEST LOSS LIVES IN A DIFFERENT VARIABLE ON EACH OF THE
                # THREE SELECTION PATHS, and comparing against the wrong one
                # would not error - the eager path's current_min_loss is
                # still 1e20 under batching, so every frame would take the
                # final pose unconditionally and the arm would silently
                # become "always commit the last pose".
                _best_so_far = (
                    _best_loss_host if es_signals.enabled
                    else float(current_min_loss_t) if _sync_free_candidates
                    else float(current_min_loss))
                _fl = float(_floss.detach())
                # Drop the graph before the next frame's tracking starts -
                # a live reference to a loss is what blocked CUDA graph
                # capture once already (see the note above the capture).
                _floss = _fv = None
                with torch.no_grad():
                    if _fl < _best_so_far:
                        candidate_cam_unnorm_rot = params[
                            'cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params[
                            'cam_trans'][..., time_idx].detach().clone()
                        _final_pose_wins += 1
                _final_pose_scored += 1
                # No backward was run, so no grads accumulated - but the
                # tracking optimiser is per-frame and about to be rebuilt
                # anyway, so nothing to clear.
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
            iter_trace.end_frame(params['cam_unnorm_rots'][..., time_idx],
                                 params['cam_trans'][..., time_idx])
            # Adam has now COMMITTED a pose for this frame. That pose is what
            # produces the reported ATE, so it - not ground truth - is the
            # reference the probe should measure against. Distance to GT
            # conflates this frame's tracking error with drift inherited from
            # earlier frames, and the drift term dominates.
            if _gn_render_fn is not None:
                with torch.no_grad():
                    _fw = torch.eye(4, device=params['means3D'].device)
                    _fw[:3, :3] = build_rotation(
                        F.normalize(candidate_cam_unnorm_rot))
                    _fw[:3, 3] = candidate_cam_tran
                gn_probe.finalize_frame(time_idx, _fw)
                # LOSS LINE-SCAN. Measures the OBJECTIVE along Adam -> ground
                # truth. Deliberately placed here, on Adam's committed pose and
                # BEFORE any GN refinement, so the curve describes the loss
                # surface the tracker actually faced rather than one GN has
                # already moved through.
                if _gn_scan_now:
                    with torch.no_grad():
                        gn_probe.line_scan(time_idx, _gn_render_fn, _fw, _gn_gt)
                # HYBRID HANDOFF: refine Adam's committed pose with GN.
                if gn_probe.use_as_tracker and gn_probe.tracker_handoff > 0:
                    _xi = gn_probe.run(time_idx, _gn_render_fn, _fw, _gn_gt,
                                       handoff=gn_probe.tracker_handoff,
                                       record=False,
                                       max_iters=gn_probe.tracker_iters,
                                       render_grad_fn=_gn_render_grad_fn)
                    if _xi is not None:
                        with torch.no_grad():
                            _new = gn_se3_exp(_xi) @ _fw
                            params['cam_unnorm_rots'][..., time_idx] =                                 gn_mat_to_quat(_new[:3, :3]).to(
                                    params['cam_unnorm_rots'].dtype).reshape(
                                    params['cam_unnorm_rots'][..., time_idx].shape)
                            params['cam_trans'][..., time_idx] = _new[:3, 3].to(
                                params['cam_trans'].dtype).reshape(
                                params['cam_trans'][..., time_idx].shape)
                    gn_probe.note_tracked(time_idx)
            # Tracking's backward populates .grad for the Gaussian attributes -
            # rgb_colors, logit_opacities and log_scales all feed rendervar and
            # require grad - but build_tracking_optimizer drops their zero-lr
            # groups, so optimizer.zero_grad() never reaches them and they
            # accumulate across every tracking iteration of the frame. Mapping
            # then builds a fresh optimizer over all params, and its first
            # step() applies that whole accumulated sum as one spurious update.
            #
            # main cannot hit this: its tracking optimizer held every group
            # (Gaussians at lr=0), so zero_grad() cleared all of them. The leak
            # arrived with the drop-zero-lr-groups optimisation.
            #
            # Off by default only so it does not silently move numbers that
            # earlier runs were measured against - it restores main's
            # semantics and should be enabled once A/B'd.
            if config['tracking'].get('clear_gaussian_grads', False):
                for _p in params.values():
                    _p.grad = None
        elif (time_idx > 0 and not config['tracking']['use_gt_poses']
                and gn_probe.use_as_tracker and gn_probe.tracker_handoff == 0
                and _gn_render_fn is not None):
            # GAUSS-NEWTON AS THE ACTUAL TRACKER. Everything before this was a
            # probe measuring alongside Adam; this replaces it, which is the
            # only way to answer the question the probe raised.
            #
            # WHAT THE PROBE ESTABLISHED: GN reaches a cost Adam never reaches,
            # in 11-17 iterations against Adam's ~70. WHAT IT DID NOT: whether
            # that lower-cost pose is a BETTER pose. GN lands 0.6-2.9 cm from
            # Adam's committed pose, and the ground-truth-error delta moved
            # toward truth on one probed frame and away on another. Only ATE
            # over a full sequence decides, and this is what produces it.
            #
            # tracker_handoff > 0 runs Adam first; 0 is pure GN. The probe
            # showed larger k does not help (k+gn simply grows), so 0 is the
            # arm to run first.
            with torch.no_grad():
                _dev = params['means3D'].device
                _b = torch.eye(4, device=_dev)
                _b[:3, :3] = build_rotation(
                    F.normalize(params['cam_unnorm_rots'][..., time_idx].detach()))
                _b[:3, 3] = params['cam_trans'][..., time_idx].detach()
            _xi = gn_probe.run(time_idx, _gn_render_fn, _b, _gn_gt,
                               handoff=0, record=False,
                               max_iters=gn_probe.tracker_iters,
                                       render_grad_fn=_gn_render_grad_fn)
            if _xi is not None:
                with torch.no_grad():
                    _new = gn_se3_exp(_xi) @ _b
                    params['cam_unnorm_rots'][..., time_idx] = gn_mat_to_quat(
                        _new[:3, :3]).to(params['cam_unnorm_rots'].dtype).reshape(
                            params['cam_unnorm_rots'][..., time_idx].shape)
                    params['cam_trans'][..., time_idx] = _new[:3, 3].to(
                        params['cam_trans'].dtype).reshape(
                            params['cam_trans'][..., time_idx].shape)
            gn_probe.note_tracked(time_idx)

        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # OPEN-LOOP TRACKING EVALUATION. Score the pose the tracker produced
        # against ground truth, then COMMIT ground truth so the map never
        # inherits tracking error and the next frame starts clean.
        #
        # This is what separates "GN overfits a good map" from "GN diverged and
        # the map followed it down" - the closed-loop run could not, because a
        # tracking failure destroys the map the objective is defined against.
        # Works for Adam too: run it both ways and the comparison is per-frame
        # error on identical maps from identical initialisations.
        if gn_probe.open_loop and time_idx > 0:
            with torch.no_grad():
                _gt = curr_gt_w2c[-1]
                _te = float((params['cam_trans'][..., time_idx].reshape(3)
                             - _gt[:3, 3]).norm() * 100.0)
                _Re = build_rotation(F.normalize(
                    params['cam_unnorm_rots'][..., time_idx].detach()))[0]
                _cos = ((_Re @ _gt[:3, :3].t()).diagonal().sum() - 1.0) / 2.0
                _de = float(torch.rad2deg(torch.acos(_cos.clamp(-1.0, 1.0))))
                gn_probe.note_open_loop(time_idx, _te, _de)
                params['cam_unnorm_rots'][..., time_idx] = matrix_to_quaternion(
                    _gt[:3, :3].unsqueeze(0).detach())
                params['cam_trans'][..., time_idx] = _gt[:3, 3].detach()

        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            n_new_pts = float('inf')  # default: no densification signal this frame
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data

                pre_num_pts = params['means3D'].shape[0]
                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, densify_curr_data,
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], config['gaussian_distribution'])
                post_num_pts = params['means3D'].shape[0]
                n_new_pts = post_num_pts - pre_num_pts
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)
            # Removes the depth loss's masked_select sync. Off by default: it
            # changes the mapping loss by floating-point summation order, so it
            # gets its own flag to be A/B'd rather than riding along.
            _map_mask_multiply = config['mapping'].get('mask_multiply_loss', False)

            # Mapping
            mapping_start_time = time.time()
            # Adaptive mapping signals — derived from n_new_pts, no extra render needed.
            # add_new_gaussians() seeds exactly in pixels where silhouette < sil_thres,
            # so n_new_pts / n_pix ≈ unseen_ratio (fraction of image not covered by Gaussians).
            #
            # unseen_ratio and n_new_pts are now TWO INDEPENDENT signals, not
            # the same number shared across both weights. unseen_ratio is
            # already pixel-normalised at the source. n_new_pts is the RAW
            # count: compute_budget() takes it in two parts, the same
            # contract Gaussian-SLAM's mapper.py already uses (see
            # AdaptiveMapper.calibrate_new_pts_frames) - n_new_pts_ratio
            # (divided by n_new_pts_ref HERE, at the current value, which is
            # the fitted one once calibration installs) drives the budget,
            # and the raw n_new_pts is recorded for the fit itself. Before,
            # reusing unseen_ratio for both made n_new_pts_ref a dead config
            # value for SplaTAM specifically - dividing it into anything.
            _n_pix = float(curr_data['im'].shape[1] * curr_data['im'].shape[2])
            _admap_unseen = 1.0  # default for first frame / no densification
            _admap_new_pts = None  # no densification signal this frame (frame 0 / inf)
            _admap_new_ratio = 1.0
            if n_new_pts != float('inf'):
                _admap_unseen = min(float(n_new_pts) / _n_pix, 1.0)
                _admap_new_pts = n_new_pts
                _admap_new_ratio = min(
                    float(n_new_pts) / max(adaptive_mapper.n_new_pts_ref, 1), 1.0)
            _admap_budget = None
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            map_trace.begin_frame(time_idx, num_iters_mapping, n_new_pts=n_new_pts,
                                  unseen_ratio=_admap_unseen)
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                # Loss for current frame
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True,
                                                mask_multiply=_map_mask_multiply)
                if map_trace.active:
                    map_trace.capture(loss.detach(), iter_time_idx)
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Adaptive mapping: lock in budget after iter 0
                    if iter == 0:
                        _admap_budget = adaptive_mapper.compute_budget(
                            unseen_ratio=_admap_unseen,
                            n_new_pts_ratio=_admap_new_ratio,
                            n_new_pts=_admap_new_pts,
                            depth_error=losses['depth'].item() if 'depth' in losses else 0.0,
                            color_error=losses['im'].item() if isinstance(losses.get('im'), torch.Tensor) else 0.0,
                        )
                        if adaptive_mapper.enabled and _admap_budget < num_iters_mapping:
                            print(f"  [AdaptiveMap] frame {time_idx}: {_admap_budget}/{num_iters_mapping} iters"
                                  f" (unseen={_admap_unseen:.3f} new_pts={_admap_new_ratio:.3f})")
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
                if _admap_budget is not None and iter + 1 >= _admap_budget:
                    break
            if num_iters_mapping > 0:
                progress_bar.close()
            map_trace.end_frame()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            # Adaptive pruning: once per frame (post-mapping), not per
            # mapping iteration - opacity/age are stable snapshots of the
            # map's current state, not something worth recomputing 30x.
            if adaptive_pruner.enabled:
                with torch.no_grad():
                    _prune_opacity = torch.sigmoid(params['logit_opacities']).squeeze(-1)
                    _prune_age = float(time_idx) - variables['timestep']
                    _prune_progress = (time_idx + 1) / num_frames
                    _prune_mask = adaptive_pruner.compute_mask(_prune_opacity, _prune_age, _prune_progress)
                    if _prune_mask.any():
                        params, variables = remove_points(_prune_mask, params, variables, optimizer)

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        
        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1

        torch.cuda.empty_cache()

    iter_trace.close()

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    print(
        "Tracking iterations/frame: "
        f"{tracking_iter_time_count / tracking_frame_time_count:.6f} "
        f"(steps={tracking_iter_time_count}, "
        f"frames={tracking_frame_time_count})"
    )
    if _tracking_post_warmup_frames > 0:
        _tracking_post_warmup_ipf = (
            _tracking_post_warmup_steps / _tracking_post_warmup_frames
        )
        print(
            "Tracking iterations/frame (post-warmup): "
            f"{_tracking_post_warmup_ipf:.6f} "
            f"(steps={_tracking_post_warmup_steps}, "
            f"frames={_tracking_post_warmup_frames}, excluded first "
            f"{_tracking_report_warmup_frames} warmup frames)"
        )
    # VERBOSE_DIAG=1 restores every summary line below, including the ones
    # for mechanisms this investigation isn't using right now (early stop,
    # adaptive pruning, the separate tracking-step graph, signal-batching
    # detail, pixel sampling, and the four probes that only ever print
    # "disabled"). Default off keeps a run's tail to the lines actually being
    # read: adaptive mapping, gradient reuse, the CUDA graph capture, the
    # windowed-convergence (incumbent) stopper, and the preconditioner.
    _verbose_diag = os.environ.get("VERBOSE_DIAG", "0") not in ("0", "", "false", "False")
    print(adaptive_mapper.summary())
    if _verbose_diag:
        print(stopper.summary())
        print(adaptive_pruner.summary())
        print(tracking_step_graph.summary())
    print(binning_capacity.summary())
    print(tracking_iteration_graph.summary())
    if _verbose_diag:
        print(es_signals.summary())
    print(windowed_stop.summary(config['tracking']['num_iters']))
    for _line in windowed_sweep.summary_lines(config['tracking']['num_iters']):
        print(_line)
    if _verbose_diag:
        print(pixel_sampler.summary())
        print(grad_var_probe.summary())
        print(tile_cost_probe.summary())
        print(stale_probe.summary())
    print(grad_reuse.summary())
    grad_var_probe.save()
    if _verbose_diag:
        print(gn_probe.summary())
    if _final_pose_scored:
        print(f"Final-pose scoring (COMMIT_AT_LOSS=2): the post-step pose beat "
              f"every scored candidate on {_final_pose_wins}/{_final_pose_scored} "
              f"frames ({100.0 * _final_pose_wins / _final_pose_scored:.0f}%)"
              + ("  <- NEVER WINS: level 2 is level 1 with a different tag"
                 if _final_pose_wins < 0.02 * _final_pose_scored else ""))
    if adam_probe is not None:
        print(adam_probe.summary(), flush=True)
    print(pose_pre.summary() if pose_pre is not None
          else "Pose preconditioner: disabled")
    if pose_pre is not None:
        _sp = pose_pre.step_profile()
        if _sp:
            print(_sp)
    if excursion_recorder is not None:
        excursion_recorder.close()
        print(excursion_recorder.summary())
    if _pre_neff_line():
        print(_pre_neff_line())
    if _verbose_diag:
        print(precond_probe.summary())
    precond_probe.save()
    gn_probe.save()
    # Read `consumer blocked` before believing any prefetch timing: a
    # prefetcher that misses every frame still runs and buys nothing.
    for _pf_ds in (dataset, densify_dataset if seperate_densification_res else None,
                   tracking_dataset if seperate_tracking_res else None):
        if hasattr(_pf_ds, 'summary'):
            print(_pf_ds.summary())
            _pf_ds.close()
    _slam_s = time.time() - slam_start_time
    print(f"SLAM-only time: {_slam_s:.1f} s (excludes final evaluation)")
    # RUN SUMMARY TO DISK. Timing was console-only, so a run whose terminal
    # scrolled lost its ms/iter, in_loop and iteration counts permanently -
    # recoverable only by repeating an hour-long run. ATE, PSNR and the rest
    # already had files; these did not.
    #
    # in_loop is computed HERE with the definition the ladder uses
    # (tracking_s_per_frame * (N-1) + mapping_s_per_frame * N) so it cannot be
    # re-derived wrongly later from a different frame count.
    try:
        _n_frames = int(num_frames)
        _summary = {
            "run_name": config["run_name"],
            "tracking_ms_per_iter": tracking_iter_time_avg * 1000,
            "tracking_s_per_frame": tracking_frame_time_avg,
            "mapping_ms_per_iter": mapping_iter_time_avg * 1000,
            "mapping_s_per_frame": mapping_frame_time_avg,
            "slam_only_s": _slam_s,
            "in_loop_s": (tracking_frame_time_avg * max(_n_frames - 1, 0)
                          + mapping_frame_time_avg * _n_frames),
            "frames": _n_frames,
            "tracking_iters_per_frame": (tracking_frame_time_avg
                                         / max(tracking_iter_time_avg, 1e-12)),
            "tracking_iters_per_frame_post_warmup": (
                _tracking_post_warmup_steps / _tracking_post_warmup_frames
                if _tracking_post_warmup_frames > 0 else None
            ),
            "tracking_post_warmup_frames": _tracking_post_warmup_frames,
            "tracking_excluded_warmup_frames": _tracking_report_warmup_frames,
            "precond_steps": (pose_pre.steps if pose_pre is not None
                              and hasattr(pose_pre, "steps") else None),
            "precond_summary": (pose_pre.summary() if pose_pre is not None
                                else None),
            "binning_summary": binning_capacity.summary(),
            "pixel_sample_summary": pixel_sampler.summary(),
            "early_stop_summary": stopper.summary(),
            "adaptive_mapping_summary": adaptive_mapper.summary(),
        }
        _sp = os.path.join(output_dir, "run_summary.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(_sp, "w") as _f:
            json.dump(_summary, _f, indent=2)
        print(f"Run summary: {_sp}")
    except Exception as _e:
        # NEVER let bookkeeping kill a finished run. An hour of GPU time is
        # already spent by this point.
        print(f"[RunSummary] could not write: {_e}")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
    
    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    # Saved BEFORE evaluation, not after. eval() only reads from params, and a
    # crash inside it would otherwise discard the entire SLAM run: a 1h37m
    # scene0059_00 run was lost exactly that way to an unguarded pose
    # inversion. With the checkpoint on disk first, evaluation can be redone
    # without re-running SLAM.
    save_params(params, output_dir)

    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)
