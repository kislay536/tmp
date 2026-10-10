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

import math
from typing import NamedTuple

import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = translate.to(R.device)
    Rt = torch.zeros((4, 4), device=R.device)
    # Rt[:3, :3] = R.transpose()
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt


def getWorld2View2_capture_safe(R, t):
    """getWorld2View2 with translate=0, scale=1 - which is every call site in
    this repo - written without the two matrix inversions.

    The original builds Rt, inverts it, adds `translate` to the resulting
    camera centre, scales it, and inverts back. With translate=(0,0,0) and
    scale=1.0 the round trip is the identity, so both inversions cancel and
    the answer is just Rt. The result is the same matrix, up to the roundoff
    the two dropped inversions were contributing.

    Why it matters here: torch.linalg.inv checks the LAPACK info flag for
    singularity, and that check copies a value device-to-host. A synchronous
    D2H is illegal while a CUDA stream is capturing, so with the original
    every render carries two capture-fatal readbacks - and camera_center's
    .inverse() a third - before the rasterizer is even reached.

    torch.eye rather than zeros + `Rt[3, 3] = 1.0`, and the difference is not
    cosmetic - the scalar assignment is capture-fatal in its own right.
    MEASURED, first real capture attempt:

        RuntimeError: CUDA error: operation not permitted when stream is
        capturing            at Rt[3, 3] = 1.0

    Assigning a PYTHON scalar into a tensor element goes through
    scalarToTensor: PyTorch materialises a CPU scalar tensor and copies it
    host-to-device, and a pageable H2D copy is synchronous. Passing a scalar to
    an OPERATOR is completely different - `tensor * 0.5`, `torch.clamp(x,
    min=1e-8)` and `tensor.fill_(1)` hand the scalar to the kernel as an
    argument and never touch host memory. So `x[i] = 1.0` is illegal under
    capture while `x.fill_(1.0)` is fine, and torch.eye is internally
    zeros + diagonal.fill_(1).

    The same trap is why `ssm[0, 1] = -x[2]` in pose_utils.skew_sym_mat is
    safe: the right-hand side there is a device tensor element, so it is a
    device-to-device copy, not a host round trip. The rule is about where the
    VALUE comes from, not about indexed assignment.
    """
    Rt = torch.eye(4, device=R.device, dtype=R.dtype)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2 * (zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
