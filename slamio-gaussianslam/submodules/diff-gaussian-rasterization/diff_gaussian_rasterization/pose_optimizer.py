#
# Shared, model-agnostic helper for building tracking-only optimizers.
#
# During pose tracking, SLAM pipelines built on this rasterizer typically
# register every Gaussian parameter (means3D, colors, opacities, scales,
# rotations, ...) alongside the camera pose in a single torch.optim.Adam,
# but set lr=0.0 on everything except the pose so only the camera moves.
# A zero learning rate makes Adam's update a mathematical no-op for that
# parameter, but Adam still runs its full elementwise update math (momentum,
# variance, bias correction) on it every step - real, launched CUDA kernels
# doing nothing. For per-Gaussian tensors spanning a large map, across
# hundreds of tracking iterations per frame, that's a lot of wasted launches.
#
# build_tracking_optimizer drops the zero-lr groups before constructing the
# optimizer instead, which is identical in every respect except that it
# doesn't do the pointless work.
#
import torch


def build_tracking_optimizer(param_groups, tracking=True, **adam_kwargs):
    """
    param_groups: list of dicts in the standard torch.optim param_group form,
        e.g. [{'params': [tensor], 'name': str, 'lr': float}, ...] - the same
        structure every model already builds for its optimizer.
    tracking: when True, groups with lr == 0.0 are dropped before the
        optimizer is constructed. When False (e.g. mapping, where the
        Gaussian parameters are the ones actually being optimized), all
        groups are kept unchanged.
    adam_kwargs: forwarded to torch.optim.Adam (e.g. amsgrad=True, eps=...).

    Returns a torch.optim.Adam equivalent in behaviour to constructing it
    with the full param_groups list, but without launching update kernels
    for parameters that can never move.
    """
    if tracking:
        param_groups = [g for g in param_groups if g.get('lr', 0.0) != 0.0]
    return torch.optim.Adam(param_groups, **adam_kwargs)
