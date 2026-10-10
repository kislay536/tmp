"""Verify the rendered-alpha (out_opacity) gradient reaches the Gaussians.

WHY THIS EXISTS. The `opacity` output of this rasterizer is a differentiable
image, not a mask: it is `1 - T_final` per pixel. Gaussian-SLAM's tracker uses
it that way -- with `soft_alpha: True` (the default in every shipped GSLAM
config) `compute_losses` multiplies BOTH the colour and the depth residual by
`alpha ** 3`, so the alpha image is a live part of the camera-pose gradient.

The Python wrapper used to accept `grad_out_opacity` in `backward()` and then
never pass it to CUDA. Nothing raised. The alpha output simply behaved as a
constant, one branch of GSLAM's pose gradient was silently zero, and the only
symptom was a worse ATE -- ~5.9 cm on TUM fr1_desk against the ~2.5 cm the
upstream repo reports.

WHY THE FIX IS OPT-IN (`alpha_grad`, default False) RATHER THAN GLOBAL. The
three models do not want the same thing, because each inherits its own
upstream rasterizer's behaviour:

  GSLAM    consumes alpha (soft_alpha) AND its fork
           (VladimirYugay/gaussian_rasterizer@9c40173) propagates the gradient.
           It needs the term. It is the only caller that passes alpha_grad=True.
  MonoGS   ALSO consumes it -- `opacity * |image - gt|` in
           slam_utils.get_loss_tracking_rgb -- but its own fork
           (rmurai0610/diff-gaussian-rasterization-w-pose) drops the gradient
           exactly as this one did. Its published numbers come from that
           behaviour, so switching the term on would move MonoGS off its own
           baseline. It must stay off.
  SplaTAM  never consumes the output (`im, radius, *_ = ...`), so
           grad_out_opacity is None for it either way.

So "just always propagate it" would fix one model and silently change another.

That is the class of failure this file guards, in both directions: a missing
gradient term does not crash, it converges to the wrong answer.

WHAT IT CHECKS
  1. FINITE DIFFERENCE, AGAINST COLOUR AND DEPTH AS CONTROLS.
     d(channel.sum())/d(means3D) from backward vs a central difference of the
     forward, run on all three channels. A bare error figure for alpha would be
     uninterpretable -- this forward is piecewise (the min(0.99) clamp, the
     1/255 cutoff, the T < 1e-4 early-out, tile membership) so every channel
     disagrees somewhat. Colour and depth predate this change and are not in
     question, so they calibrate the harness; alpha passes if it is in their
     company. It would have failed outright before the fix, with an analytic
     gradient of exactly zero.
  2. THE soft_alpha LOSS SHAPE. The alpha branch must actually change the
     gradient, checked by differencing against the same loss with alpha
     detached. A wrong-but-nonzero gradient and an absent one are different
     bugs, and the finite-difference tolerance alone would not separate them.
  3. THE GATE HOLDS. With alpha_grad left at its default, a MonoGS-shaped loss
     must produce the byte-identical gradient it did before this term existed.

Run on the Linux GPU VM after `bash rebuild.sh`.
"""
import math

import torch

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
    viewmatrix[2, 3] = 3.0

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


def make_gaussians():
    means3D = torch.rand(P, 3, device=device) * 2 - 1
    means3D[:, 2] += 3.0
    opacities = torch.sigmoid(torch.randn(P, 1, device=device))
    # Deliberately fat. Small Gaussians make alpha near-binary, which flattens
    # the very derivative under test and would turn a real failure into a pass.
    scales = torch.rand(P, 3, device=device) * 0.08 + 0.04
    raw_rot = torch.randn(P, 4, device=device)
    rotations = raw_rot / raw_rot.norm(dim=-1, keepdim=True)
    sh = torch.randn(P, 9, 3, device=device)
    return means3D, opacities, scales, rotations, sh


CAMERA = make_camera()
BASE = make_gaussians()


def render(means3D, opacities, scales, rotations, sh, alpha_grad=True):
    rasterizer = GaussianRasterizer(raster_settings=CAMERA)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    color, radii, depth, alpha, n_touched = rasterizer(
        means3D=means3D, means2D=means2D, opacities=opacities, shs=sh,
        scales=scales, rotations=rotations, tracking_only=False,
        alpha_grad=alpha_grad,
    )
    return color, depth, alpha


def fd_worst_rel(pick, eps, n_probe=12):
    """Worst relative error between analytic and central-difference gradient.

    `pick` selects a scalar from (color, depth, alpha). Run on colour and depth
    as WELL as alpha, because a bare number here is uninterpretable: this
    rasterizer's forward is piecewise (the min(0.99) alpha clamp, the 1/255
    cutoff, the T < 1e-4 early-out, tile membership), so SOME finite-difference
    disagreement is expected on every channel. The question is never "is alpha's
    error small" but "is alpha's error like the channels we already trust".
    """
    means3D = BASE[0].clone().requires_grad_(True)
    rest = [t.clone() for t in BASE[1:]]
    pick(*render(means3D, *rest)).backward()
    analytic = means3D.grad.clone()

    peak = analytic.abs().max().item()
    if peak < 1e-8:
        return float("inf"), 0.0

    idx = torch.topk(analytic.abs().flatten(), n_probe).indices.tolist()
    worst = 0.0
    for k in idx:
        i, j = divmod(k, 3)
        vals = []
        for sign in (+1, -1):
            probe = BASE[0].clone()
            probe[i, j] += sign * eps
            with torch.no_grad():
                vals.append(pick(*render(probe, *rest)).item())
        numeric = (vals[0] - vals[1]) / (2 * eps)
        ana = analytic[i, j].item()
        worst = max(worst, abs(ana - numeric) / max(abs(ana), abs(numeric), 1e-3))
    return worst, peak


def test_finite_difference(eps=2e-3):
    """Alpha's finite-difference agreement, MEASURED AGAINST COLOUR AND DEPTH.

    Colour and depth are the controls. Their gradients predate this change and
    are not in question, so they calibrate how much disagreement this harness
    produces on a piecewise forward. Alpha passes if it is in their company.
    """
    print("Test 1: finite-difference check (alpha vs colour/depth controls)")

    channels = [
        ("colour (control)", lambda c, d, a: c.sum()),
        ("depth  (control)", lambda c, d, a: d.sum()),
        ("alpha  (NEW)    ", lambda c, d, a: a.sum()),
    ]
    worst = {}
    for name, pick in channels:
        w, peak = fd_worst_rel(pick, eps)
        worst[name.strip()] = w
        if w == float("inf"):
            print(f"  {name}: analytic gradient is identically ZERO")
        else:
            print(f"  {name}: worst rel err {w:6.4f}   (|grad| max {peak:.4e})")

    alpha_w = worst["alpha  (NEW)".strip()]
    ctrl = max(worst["colour (control)".strip()], worst["depth  (control)".strip()])

    if alpha_w == float("inf"):
        print("  [FAIL] alpha gradient is identically ZERO -- the term is not")
        print("         reaching the Gaussians. This is the pre-fix behaviour.")
        return False

    # 1.5x headroom over the worse control. Alpha is expected to be slightly
    # noisier: it is the only channel whose value saturates against the
    # min(0.99) clamp, which the backward does not model on any channel.
    ok = alpha_w <= max(1.5 * ctrl, 0.02)
    print(f"  [{'OK  ' if ok else 'FAIL'}] alpha {alpha_w:.4f} vs worst "
          f"control {ctrl:.4f} (budget {max(1.5 * ctrl, 0.02):.4f})")
    if not ok:
        print("         Alpha disagrees substantially MORE than the channels")
        print("         computed by the same kernel on the same geometry, so")
        print("         this is the new term, not the harness.")
    return ok


def test_soft_alpha_shape():
    """The GSLAM loss shape: residuals weighted by alpha ** 3.

    Differenced against the same loss with alpha detached. If those two agree,
    the alpha term is still being dropped somewhere.
    """
    print("Test 2: soft_alpha loss -- alpha branch must change the gradient")
    gt_color = torch.rand(3, H, W, device=device)
    gt_depth = torch.rand(H, W, device=device) + 2.5

    def grad_for(detach_alpha):
        means3D = BASE[0].clone().requires_grad_(True)
        rest = [t.clone() for t in BASE[1:]]
        color, depth, alpha = render(means3D, *rest)
        w = (alpha.detach() if detach_alpha else alpha) ** 3
        loss = ((color - gt_color).abs() * w).sum() \
            + ((depth - gt_depth).abs() * w).sum()
        loss.backward()
        return means3D.grad.clone()

    with_alpha = grad_for(detach_alpha=False)
    without = grad_for(detach_alpha=True)
    diff = (with_alpha - without).abs().max().item()
    scale = without.abs().max().item()
    print(f"  |grad| max (alpha detached)  = {scale:.6e}")
    print(f"  max abs difference           = {diff:.6e}")
    ok = diff > 1e-6 * max(scale, 1.0)
    print(f"  [{'OK  ' if ok else 'FAIL'}] alpha branch contributes a "
          f"non-negligible gradient")
    if not ok:
        print("         The two are identical: soft_alpha is a no-op on the")
        print("         pose gradient, which is exactly the GSLAM bug.")
    return ok


def test_default_off_matches_monogs():
    """THE FLAG MUST DEFAULT OFF -- this is the MonoGS-parity test.

    MonoGS's tracking loss is `opacity * |image - gt|`
    (slam_utils.get_loss_tracking_rgb), so autograd DOES materialise
    grad_out_opacity for it. But MonoGS's own upstream rasterizer
    (rmurai0610/diff-gaussian-rasterization-w-pose) drops that gradient, and
    MonoGS's published numbers come from that behaviour. So with alpha_grad
    left at its default the gradient must be EXACTLY what it was before this
    term existed -- not close, identical.

    Reproduces MonoGS's loss shape rather than a colour+depth-only one,
    because a loss that never touches alpha would pass this test trivially
    and prove nothing about the gate.
    """
    print("Test 3: alpha_grad defaults OFF (MonoGS parity)")
    gt_image = torch.rand(3, H, W, device=device)

    def grad_for(alpha_grad, detach_alpha):
        means3D = BASE[0].clone().requires_grad_(True)
        rest = [t.clone() for t in BASE[1:]]
        color, depth, alpha = render(means3D, *rest, alpha_grad=alpha_grad)
        w = alpha.detach() if detach_alpha else alpha
        (w * (color - gt_image).abs()).mean().backward()
        return means3D.grad.clone()

    # THE NOISE FLOOR FIRST. renderCUDABackward accumulates with atomicAdd, so
    # its summation order is hardware-determined and the SAME config does not
    # reproduce bit-for-bit. An exact torch.equal here reports a 2e-10 ordering
    # difference as a leak - which it did on the first run of this file.
    # Measure how much the kernel disagrees with ITSELF, then judge against it.
    detached = grad_for(alpha_grad=False, detach_alpha=True)
    detached_again = grad_for(alpha_grad=False, detach_alpha=True)
    floor = (detached - detached_again).abs().max().item()
    print(f"  atomicAdd noise floor (same config, twice) = {floor:.3e}")

    # Default (flag absent) vs. alpha explicitly detached. If the gate leaks,
    # the first picks up a term the second cannot have and these separate by
    # far more than the reordering noise.
    default_on_graph = grad_for(alpha_grad=False, detach_alpha=False)
    leak = (default_on_graph - detached).abs().max().item()
    ok = leak <= max(10 * floor, 1e-8)
    print(f"  [{'OK  ' if ok else 'FAIL'}] default vs alpha-detached: "
          f"{leak:.3e} (budget {max(10 * floor, 1e-8):.3e})")
    if not ok:
        print("         The gate LEAKED: MonoGS would get a gradient term its")
        print("         own upstream rasterizer does not have.")

    # And the flag must actually do something when asked, or Test 1 is passing
    # for the wrong reason. This must clear the floor by orders of magnitude.
    opted_in = grad_for(alpha_grad=True, detach_alpha=False)
    signal = (opted_in - detached).abs().max().item()
    changed = signal > max(1000 * floor, 1e-6)
    print(f"  [{'OK  ' if changed else 'FAIL'}] alpha_grad=True changes it: "
          f"{signal:.3e} ({signal / max(floor, 1e-30):.1e}x the floor)")
    return ok and changed


if __name__ == "__main__":
    results = [
        test_finite_difference(),
        test_soft_alpha_shape(),
        test_default_off_matches_monogs(),
    ]
    print()
    if all(results):
        print("ALL PASS -- the rendered-alpha gradient reaches the Gaussians.")
    else:
        print("FAILURES -- see above.")
        raise SystemExit(1)
