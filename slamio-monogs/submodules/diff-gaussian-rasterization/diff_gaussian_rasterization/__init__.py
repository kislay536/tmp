#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import os
import torch.nn as nn
import torch
from . import _C
from .pose_optimizer import build_tracking_optimizer

# Whether a masked forward also masks its BACKWARD render.
#
# A RUNTIME switch, not a build flag like DGR_WARP_REDUCE, and on purpose: the
# two arms of this A/B are then the SAME .so, so the standing "which arm was
# which" hazard cannot apply to it and there is no per-arm rebuild to get wrong
# in an env that installs the rasterizer three times.
#
#   DGR_MASK_BACKWARD=1  (default)  mask both halves
#   DGR_MASK_BACKWARD=0             mask the forward only - reproduces the
#                                   behaviour every sparse number on this
#                                   branch was measured against
#
# The gradients are identical either way; see renderCUDABackward. This moves
# time, not results, which is what makes the A/B a pure timing comparison.
_MASK_BACKWARD = os.environ.get("DGR_MASK_BACKWARD", "1") not in ("0", "false", "False")

_MASK_BACKWARD_LOGGED = False
def _log_mask_backward_once():
    global _MASK_BACKWARD_LOGGED
    if not _MASK_BACKWARD_LOGGED:
        _MASK_BACKWARD_LOGGED = True
        print(f"[diff_gaussian_rasterization] mask_backward={_MASK_BACKWARD} "
              f"(first masked render; DGR_MASK_BACKWARD to change)", flush=True)

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

# One-time diagnostic: prints the first time each tracking_only value is seen,
# so it's easy to confirm from a real training/tracking log whether the flag
# is actually reaching the rasterizer (and which __init__.py is loaded).
_TRACKING_ONLY_LOGGED = set()
def _log_tracking_only_once(tracking_only):
    if tracking_only not in _TRACKING_ONLY_LOGGED:
        _TRACKING_ONLY_LOGGED.add(tracking_only)
        print(f"[diff_gaussian_rasterization] tracking_only={tracking_only} "
              f"(first occurrence, module={__file__})", flush=True)


def build_variant():
    """Which compile-time variants this .so was built with.

    The BUILD prints its flags; a RUN did not, so a measurement could be
    attributed to the wrong binary just by losing track of which build was
    installed. A SplaTAM A/B came out 11.3% apart and the first question asked
    was "which arm was which" - which is a question a log line should answer.
    """
    try:
        v = {"warp_reduce": _C.warp_reduce_enabled(),
             "fast_exp": _C.fast_exp_enabled()}
    except AttributeError:
        # Older .so without the introspection binding - rebuild to get it.
        return {"warp_reduce": None, "fast_exp": None, "min_blocks_per_sm": None}
    # Added later than the other two, so a .so can have those and not this.
    # Report it as 0 (the default, unconstrained form) rather than None, since
    # every build predating the flag genuinely was that form.
    try:
        v["min_blocks_per_sm"] = _C.min_blocks_per_sm()
    except AttributeError:
        v["min_blocks_per_sm"] = 0
    # Tile dims, added later still. Default to 16x16 rather than None: every
    # build predating the flag genuinely was 16x16, and callers size real
    # buffers from this (utils/pixel_sample.py), so None would be worse than a
    # correct historical value.
    try:
        v["block_x"] = _C.block_x()
        v["block_y"] = _C.block_y()
    except AttributeError:
        v["block_x"] = 16
        v["block_y"] = 16
    # Ceiling-probe builds produce garbage gradients. Default False so an older
    # .so is not accused of being one.
    try:
        v["stub_grads"] = _C.stub_grads()
    except AttributeError:
        v["stub_grads"] = False
    return v


def tile_size():
    """(BLOCK_X, BLOCK_Y) the installed extension was compiled with.

    Anything building a per-tile structure must call this instead of assuming
    16. A mismatch does not raise - it silently produces a mask of the wrong
    length for the grid.
    """
    v = build_variant()
    return v["block_x"], v["block_y"]


_BUILD_LOGGED = False
def _log_build_once():
    global _BUILD_LOGGED
    if _BUILD_LOGGED:
        return
    _BUILD_LOGGED = True
    v = build_variant()
    if v["warp_reduce"] is None:
        print("[diff_gaussian_rasterization] build variant UNKNOWN - this .so "
              "predates the introspection binding; rebuild to identify it",
              flush=True)
    else:
        mb = v["min_blocks_per_sm"]
        mb_s = "default" if not mb else str(mb)
        print(f"[diff_gaussian_rasterization] build: "
              f"warp_reduce={v['warp_reduce']}  fast_exp={v['fast_exp']}  "
              f"min_blocks_per_sm={mb_s}  "
              # Tile size is a build variant too, so it belongs in the line the
              # A/B driver asserts against. Without it a stale 16x16 wheel
              # passes as an 8x8 arm - and pip HAS served a stale wheel on
              # unchanged source before, because an env var is not part of its
              # cache key. That is the exact failure this line exists to catch.
              f"tile={v['block_x']}x{v['block_y']}",
              flush=True)
        if v.get("stub_grads"):
            # Loud, every run, unmissable. The failure mode is SILENT: a stub
            # build converges to nonsense and still prints a plausible ATE.
            bar = "!" * 70
            for line in (
                bar,
                "!! DGR_STUB_GRADS BUILD - renderCUDABackward DOES NOT",
                "!! ACCUMULATE GRADIENTS. EVERY GRADIENT IS WRONG.",
                "!! Kernel timing only. Any ATE/PSNR/loss here is garbage.",
                bar,
            ):
                print(line, flush=True)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    theta,
    rho,
    raster_settings,
    tracking_only=False,
    skip_color_grad=False,
    alpha_grad=False,
    tile_mask=None,
    binning_capacity=-1,
    binning_overflow=None,
    binning_count_out=None,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
        tracking_only,
        skip_color_grad,
        alpha_grad,
        tile_mask,
        binning_capacity,
        binning_overflow,
        binning_count_out,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
        tracking_only,
        skip_color_grad,
        alpha_grad,
        tile_mask,
        binning_capacity,
        binning_overflow,
        binning_count_out,
    ):
        _log_build_once()
        _log_tracking_only_once(tracking_only)

        # Let unused outputs (depth/opacity when a caller only consumes
        # color, e.g. SplaTAM) come back as None in backward() instead of
        # a materialized zero tensor, so we can skip computing their
        # gradients in CUDA entirely rather than compute-and-discard them.
        ctx.set_materialize_grads(False)

        # Restructure arguments the way that the C++ lib expects them
        compute_n_touched = theta is not None and theta.numel() > 0
        _tile_mask = tile_mask if tile_mask is not None else torch.Tensor([])
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            _tile_mask,
            compute_n_touched,
            False,  # collect_stats — set to True externally to enable counter
            binning_capacity if binning_capacity is not None else -1,
            binning_overflow if binning_overflow is not None else torch.Tensor(),
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth, opacity, n_touched, stats = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth, opacity, n_touched, stats = _C.rasterize_gaussians(*args)

        # Surface the instance count the C++ side computed. Callers sizing a
        # fixed binning capacity need it, and returning it would change the
        # tuple every model unpacks, so it is written into a caller-owned
        # tensor instead. Warm-up only - on the capacity path num_rendered is
        # the capacity the caller already chose.
        if binning_count_out is not None:
            binning_count_out.fill_(int(num_rendered))

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.compute_pose_grad = theta is not None and theta.numel() > 0
        ctx.tracking_only = tracking_only
        ctx.skip_color_grad = skip_color_grad
        ctx.alpha_grad = alpha_grad
        # _tile_mask, not tile_mask: already normalised to a tensor above, so
        # backward needs no None handling. Saved rather than recomputed because
        # the backward's early return is only gradient-neutral against THIS
        # forward's mask - a mask rebuilt in backward would be a different
        # random draw (build_tile_mask's uniform half uses randperm).
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, _tile_mask)
        return color, radii, depth, opacity, n_touched

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_radii, grad_out_depth, grad_out_opacity, grad_n_touched):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, tile_mask = ctx.saved_tensors

        # Restructure args as C++ method expects them
        compute_pose_grad = ctx.compute_pose_grad
        tracking_only = ctx.tracking_only
        skip_color_grad = ctx.skip_color_grad
        alpha_grad = ctx.alpha_grad

        # grad_out_color USED TO BE ASSUMED ALWAYS MATERIALIZED - "color is
        # always consumed by callers" - and for all three models it is. That
        # assumption only became reachable-false once the alpha output got its
        # own gradient path: a backward through alpha ALONE leaves color unused,
        # so autograd hands us None here and pybind11 rejects it as arg13.
        #
        # It cannot be replaced with an empty tensor the way the others are:
        # RasterizeGaussiansBackwardCUDA derives H and W from
        # dL_dout_color.size(1)/size(2), so an empty one collapses the image
        # dimensions. Zeros of the right shape are the correct neutral element
        # - the kernel reads dL_dpixels for the bg_dot_dpixel term on every
        # Gaussian regardless of whether the caller wanted a colour gradient.
        #
        # Costs nothing in practice: no model reaches this branch, only an
        # alpha-only backward such as tests/test_alpha_grad.py does.
        if grad_out_color is None:
            grad_out_color = means3D.new_zeros(
                (3, raster_settings.image_height, raster_settings.image_width))

        # grad_out_depth comes back as None when the caller never used the
        # `depth` output (e.g. SplaTAM, which packs depth into colors_precomp
        # instead) - in that case skip the depth-gradient machinery in CUDA
        # entirely by passing an empty tensor, matching the original
        # rasterizer's zero depth-backward cost for those callers.
        if grad_out_depth is None:
            grad_out_depth = torch.Tensor([])

        # THE ALPHA GRADIENT IS OPT-IN PER CALL. It is NOT gated on whether
        # autograd materialised grad_out_opacity, and that distinction is the
        # whole point of this flag.
        #
        # grad_out_opacity was previously accepted here and never passed to
        # CUDA, so the `opacity` output behaved as a constant regardless of
        # what the loss did with it. Two of the three models WANT that:
        #
        #   MonoGS   consumes opacity differentiably - its tracking loss is
        #            `opacity * |image - gt|` (slam_utils.get_loss_tracking_rgb)
        #            - but its OWN upstream rasterizer
        #            (rmurai0610/diff-gaussian-rasterization-w-pose) drops
        #            grad_out_opacity in exactly the same way. Turning the term
        #            on for MonoGS would make it diverge from the code its
        #            published numbers came from, so it stays OFF by default and
        #            MonoGS does not pass this flag.
        #   SplaTAM  never consumes the output at all (`im, radius, *_ = ...`,
        #            silhouette packed into colors_precomp), so grad_out_opacity
        #            is None for it either way.
        #
        #   GSLAM    consumes it AND its upstream fork
        #            (VladimirYugay/gaussian_rasterizer@9c40173) propagates it:
        #            `dL_dalpha += (1 - accum_rea) * dL_dpixel_alpha`. soft_alpha
        #            weights both tracking residuals by alpha**3, so dropping the
        #            term zeroed a live branch of its pose gradient. That is why
        #            GSLAM's TUM fr1_desk ATE sat at ~5.9cm against ~2.5cm.
        #
        # So the flag reproduces EACH model's own upstream, which a single
        # global default cannot do.
        if not alpha_grad or grad_out_opacity is None:
            grad_out_opacity = torch.Tensor([])

        # An empty tensor here means a DENSE backward, which is what every
        # sparse measurement on this branch so far actually ran: the mask was
        # plumbed into the forward render and nowhere else, so a masked tile
        # still paid its full backward range in memory traffic and barriers.
        if tile_mask.numel() > 0:
            _log_mask_backward_once()
            if not _MASK_BACKWARD:
                tile_mask = torch.Tensor([])

        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.projmatrix_raw,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_out_depth,
                grad_out_opacity,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug,
                compute_pose_grad,
                tracking_only,
                skip_color_grad,
                tile_mask)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_tau = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_tau = _C.rasterize_gaussians_backward(*args)
        
        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        grad_rho = grad_tau[:3].view(1, -1)
        grad_theta = grad_tau[3:].view(1, -1)


        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_theta,
            grad_rho,
            None,  # raster_settings
            None,  # tracking_only - bool
            None,  # skip_color_grad - bool. MUST be here: autograd requires one
                   # entry per forward() input, and forward gained this one.
                   # Omitting it does not fail loudly at the definition - it
                   # raises at the first backward, mid-run.
            None,  # alpha_grad - bool, same rule as skip_color_grad above
            None,  # tile_mask - not differentiable
            None,  # binning_capacity - host scalar
            None,  # binning_overflow - device flag, not differentiable
            None,  # binning_count_out - measurement sink, not differentiable
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    projmatrix_raw : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, theta=None, rho=None, tracking_only=False, skip_color_grad=False, alpha_grad=False, tile_mask=None, binning_capacity=-1, binning_overflow=None, binning_count_out=None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if theta is None:
            theta = torch.Tensor([])
        if rho is None:
            rho = torch.Tensor([])
        

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            theta,
            rho,
            raster_settings,
            tracking_only,
            skip_color_grad,
            alpha_grad,
            tile_mask,
            binning_capacity,
            binning_overflow,
            binning_count_out,
        )

