"""
Regression test for the `tracking_only` backward flag.

`tracking_only=True` is supposed to be pure dead-code elimination: it must
never change any gradient that is actually load-bearing for camera-pose
optimization. This script runs backward twice on identical inputs (once with
tracking_only=False, once True) for both pose-gradient mechanisms used in
this repo:

  - "MonoGS-style": theta/rho passed -> dL_dtau active. grad_means2D and
    grad_tau (the pose gradient) must match exactly; grad_opacities and
    grad_sh must be exactly zero under tracking_only=True (proving the skip
    fired) while nonzero under tracking_only=False; grad_scales/grad_rotations
    must also be zeroed under tracking_only=True since dL_dtau supersedes them.

  - "SplaTAM-style": no theta/rho -> dL_dtau inactive, pose gradient instead
    reaches means3D/rotations via a differentiable reparameterization.
    grad_means2D, grad_means3D, grad_rotations, grad_cov3Ds_precomp must all
    match exactly (never touched by tracking_only when dL_dtau is nullptr);
    only grad_opacities/grad_sh should be zeroed.

Run on the Linux GPU VM after `pip install -e .` in this directory.
"""
import torch
import math

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

torch.manual_seed(0)
device = "cuda"
P = 200
H, W = 64, 64


def make_camera():
    fovx = fovy = math.radians(60)
    tanfovx, tanfovy = math.tan(fovx / 2), math.tan(fovy / 2)
    viewmatrix = torch.eye(4, device=device)
    viewmatrix[2, 3] = 3.0  # push camera back so points are in front of it

    proj = torch.zeros(4, 4, device=device)
    znear, zfar = 0.01, 100.0
    proj[0, 0] = 1.0 / tanfovx
    proj[1, 1] = 1.0 / tanfovy
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    proj[3, 2] = 1.0
    full_proj = (viewmatrix.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    return GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix.transpose(0, 1).contiguous(),
        projmatrix=full_proj.transpose(0, 1).contiguous(),
        projmatrix_raw=proj.transpose(0, 1).contiguous(),
        sh_degree=2,
        campos=torch.tensor([0.0, 0.0, -3.0], device=device),
        prefiltered=False,
        debug=False,
    )


def make_gaussians(requires_grad):
    means3D = (torch.rand(P, 3, device=device) * 2 - 1).requires_grad_(requires_grad)
    means3D.data[:, 2] += 3.0  # ensure in front of camera (view space z > 0)
    opacities = torch.sigmoid(torch.randn(P, 1, device=device)).requires_grad_(requires_grad)
    scales = (torch.rand(P, 3, device=device) * 0.05 + 0.01).requires_grad_(requires_grad)
    raw_rot = torch.randn(P, 4, device=device)
    rotations = (raw_rot / raw_rot.norm(dim=-1, keepdim=True)).requires_grad_(requires_grad)
    sh = torch.randn(P, 9, 3, device=device).requires_grad_(requires_grad)
    return means3D, opacities, scales, rotations, sh


def clone_leaf(t):
    return t.detach().clone().requires_grad_(True)


def close(a, b, name):
    ok = torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name}: max abs diff = {(a - b).abs().max().item():.3e}")
    return ok


def scenario_monogs():
    print("Scenario A: MonoGS-style (theta/rho active, dL_dtau path)")
    camera = make_camera()
    base = make_gaussians(requires_grad=True)

    results = {}
    for tracking_only in (False, True):
        means3D, opacities, scales, rotations, sh = [clone_leaf(t) for t in base]
        theta = torch.zeros(1, 3, device=device, requires_grad=True)
        rho = torch.zeros(1, 3, device=device, requires_grad=True)

        rasterizer = GaussianRasterizer(raster_settings=camera)
        means2D = torch.zeros_like(means3D, requires_grad=True)
        means2D.retain_grad()
        color, radii, depth, alpha, n_touched = rasterizer(
            means3D=means3D, means2D=means2D, opacities=opacities, shs=sh,
            scales=scales, rotations=rotations, theta=theta, rho=rho,
            tracking_only=tracking_only,
        )
        (color.sum() + depth.sum()).backward()

        results[tracking_only] = dict(
            means2D=means2D.grad.clone(),
            theta=theta.grad.clone(),
            rho=rho.grad.clone(),
            opacities=opacities.grad.clone(),
            scales=scales.grad.clone(),
            rotations=rotations.grad.clone(),
            sh=sh.grad.clone(),
        )

    off, on = results[False], results[True]
    ok = True
    ok &= close(off["means2D"], on["means2D"], "grad_means2D")
    ok &= close(off["theta"], on["theta"], "grad_theta (pose)")
    ok &= close(off["rho"], on["rho"], "grad_rho (pose)")

    zero = torch.zeros_like(on["opacities"])
    ok &= close(on["opacities"], zero, "grad_opacities under tracking_only (must be zero)")
    ok &= close(on["scales"], torch.zeros_like(on["scales"]), "grad_scales under tracking_only (must be zero)")
    ok &= close(on["rotations"], torch.zeros_like(on["rotations"]), "grad_rotations under tracking_only (must be zero)")
    ok &= close(on["sh"], torch.zeros_like(on["sh"]), "grad_sh under tracking_only (must be zero)")

    nonzero = off["opacities"].abs().max().item() > 1e-8
    print(f"  [{'OK  ' if nonzero else 'FAIL'}] grad_opacities nonzero under tracking_only=False (sanity: {off['opacities'].abs().max().item():.3e})")
    ok &= nonzero
    return ok


def scenario_splatam():
    print("Scenario B: SplaTAM-style (no theta/rho, reparam-through-means3D path)")
    camera = make_camera()
    base_means3D, opacities0, scales0, rotations0, sh0 = make_gaussians(requires_grad=False)

    results = {}
    for tracking_only in (False, True):
        opacities = clone_leaf(opacities0)
        scales = clone_leaf(scales0)
        rotations = clone_leaf(rotations0)
        sh = clone_leaf(sh0)

        cam_trans = torch.zeros(3, device=device, requires_grad=True)
        means3D = base_means3D.detach() + cam_trans  # stand-in for transform_to_frame

        rasterizer = GaussianRasterizer(raster_settings=camera)
        means2D = torch.zeros_like(means3D, requires_grad=True)
        means2D.retain_grad()
        color, radii, depth, alpha, n_touched = rasterizer(
            means3D=means3D, means2D=means2D, opacities=opacities, shs=sh,
            scales=scales, rotations=rotations, theta=None, rho=None,
            tracking_only=tracking_only,
        )
        (color.sum() + depth.sum()).backward()

        results[tracking_only] = dict(
            means2D=means2D.grad.clone(),
            cam_trans=cam_trans.grad.clone(),
            rotations=rotations.grad.clone(),
            opacities=opacities.grad.clone(),
            sh=sh.grad.clone(),
        )

    off, on = results[False], results[True]
    ok = True
    ok &= close(off["means2D"], on["means2D"], "grad_means2D")
    ok &= close(off["cam_trans"], on["cam_trans"], "grad_cam_trans (pose, via means3D)")
    ok &= close(off["rotations"], on["rotations"], "grad_rotations (not skipped: dL_dtau nullptr)")

    ok &= close(on["opacities"], torch.zeros_like(on["opacities"]), "grad_opacities under tracking_only (must be zero)")
    ok &= close(on["sh"], torch.zeros_like(on["sh"]), "grad_sh under tracking_only (must be zero)")

    nonzero = off["opacities"].abs().max().item() > 1e-8
    print(f"  [{'OK  ' if nonzero else 'FAIL'}] grad_opacities nonzero under tracking_only=False (sanity: {off['opacities'].abs().max().item():.3e})")
    ok &= nonzero
    return ok


if __name__ == "__main__":
    ok_a = scenario_monogs()
    print()
    ok_b = scenario_splatam()
    print()
    if ok_a and ok_b:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        raise SystemExit(1)
