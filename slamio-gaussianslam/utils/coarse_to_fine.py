import torch
import torch.nn.functional as F
import numpy as np


def scale_intrinsics(intrinsics, scale):
    """Return a copy of 3×3 intrinsics matrix scaled for `scale`× downsampling."""
    k = np.array(intrinsics, dtype=np.float64)
    k[0, 0] /= scale   # fx
    k[0, 2] /= scale   # cx
    k[1, 1] /= scale   # fy
    k[1, 2] /= scale   # cy
    return k


def downsample_image(im, scale):
    """Bilinear downsample a (C, H, W) tensor by integer scale."""
    h, w = im.shape[-2] // scale, im.shape[-1] // scale
    return F.interpolate(
        im.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
    ).squeeze(0)


def downsample_depth(depth, scale):
    """Nearest-neighbor downsample a (H, W) or (1, H, W) depth tensor by integer scale.

    Nearest-neighbor preserves valid/invalid depth boundaries, avoiding
    interpolated values between foreground and background depths.
    """
    squeezed = depth.dim() == 2
    if squeezed:
        depth = depth.unsqueeze(0)
    h, w = depth.shape[-2] // scale, depth.shape[-1] // scale
    out = F.interpolate(depth.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)
    return out.squeeze(0) if squeezed else out


def get_levels(cfg):
    """Normalize config to a list of (scale, ratio) pairs, coarsest first.

    Supports both pyramid format:
        levels: [{scale: 4, ratio: 0.25}, {scale: 2, ratio: 0.25}]
    and legacy single-scale format:
        scale: 2, coarse_ratio: 0.5
    """
    if 'levels' in cfg:
        return [(int(l['scale']), float(l['ratio'])) for l in cfg['levels']]
    if 'scale' in cfg:
        return [(int(cfg['scale']), float(cfg.get('coarse_ratio', 0.5)))]
    return []


def get_iter_scale(iter_idx, num_iters, cfg):
    """Return the downsampling scale to use at this iteration (1 = full resolution).

    Iterates through levels in order, each level consuming `ratio` fraction of
    total iterations. Remaining iterations always run at full resolution.
    """
    if not cfg.get('enabled', False) or num_iters == 0:
        return 1
    frac = iter_idx / num_iters
    cumulative = 0.0
    for scale, ratio in get_levels(cfg):
        cumulative += ratio
        if frac < cumulative:
            return scale
    return 1


def coarse_iter_count(num_iters, cfg):
    """Legacy helper — total number of sub-full-resolution iterations."""
    if not cfg.get('enabled', False):
        return 0
    total = sum(ratio for _, ratio in get_levels(cfg))
    return max(0, int(num_iters * min(total, 1.0)))
