import numpy as np
import torch


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


# `angle` is a device tensor, so `if angle < 1e-5` in the two functions below
# was a host sync AND data-dependent control flow - two per update_pose, so two
# per tracking iteration. Both are capture-fatal, and the control-flow half is
# the nastier one: a capture would bake in whichever branch the capture
# iteration happened to take and replay it for the rest of the frame. Since a
# frame's tracking converges, the capture iteration usually sits near the
# small-angle threshold, so which branch got frozen would vary by frame.
#
# torch.where evaluates both branches and selects elementwise, giving the same
# value with no branch. The clamp is what keeps that safe: the large-angle
# expressions divide by angle, angle**2 and angle**3, so at angle=0 they would
# produce NaN, and while torch.where discards a NaN in the UNSELECTED forward
# branch it does not discard it in the backward (where's grad is grad*mask, and
# NaN*0 is NaN). update_pose runs under no_grad today so no backward reaches
# here, but clamping costs nothing and removes the trap rather than relying on
# a caller staying inside no_grad forever.
_SMALL_ANGLE = 1e-5


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)

    small = angle < _SMALL_ANGLE
    a = torch.clamp(angle, min=1e-8)
    taylor = I + W + 0.5 * W2
    exact = I + (torch.sin(a) / a) * W + ((1 - torch.cos(a)) / (a**2)) * W2
    return torch.where(small, taylor, exact)


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)

    small = angle < _SMALL_ANGLE
    a = torch.clamp(angle, min=1e-8)
    taylor = I + 0.5 * W + (1.0 / 6.0) * W2
    exact = (
        I
        + W * ((1.0 - torch.cos(a)) / (a**2))
        + W2 * ((a - torch.sin(a)) / (a**3))
    )
    return torch.where(small, taylor, exact)


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(camera, converged_threshold=1e-4):
    """Fold the optimised pose delta into the camera pose.

    Entirely device-side and capture-safe as written: `converged` is returned
    as a 0-dim device tensor, not a Python bool, so nothing here synchronises.
    The sync is at the CALLER's `if converged`, which is where it has to be
    dealt with - see the frontend, which routes it through ESSignalBuffer under
    the iteration graph instead of reading it every iteration.
    """
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    # A caller passing a negative threshold (DISABLE_NATIVE_CONV, see
    # slam_frontend.py's self.native_conv_threshold) gets `converged`
    # permanently False: tau.norm() is a norm, never negative.
    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged
