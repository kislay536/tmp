"""Verify that masking the BACKWARD render changes timing and not gradients.

WHY THIS EXISTS. Sparse tile sampling builds a per-tile keep mask and hands it
to the rasterizer. Until now that mask reached `renderCUDA` and nothing else:
`renderCUDABackward` had no tile_mask parameter at all. The comment in
forward.cu explained why it did not need one, and the ARITHMETIC half of that
explanation is correct -- a masked tile has n_contrib = 0 written for every one
of its pixels, so `last_contributor` is 0 in the backward, so `skip`/`active`
is false for every Gaussian, and every write in that kernel is gated on it.

But the predicate gates the MATH, not the LOOP. A masked tile still ran its
full Gaussian range: the collective global->shared load of point_list, means2D,
conic_opacity, colors and depths, one block.sync() per round, and a
__syncthreads_count or __ballot_sync per Gaussian to discover there was nothing
to do. So sparse was buying forward-render time against a backward bill that
grew with map density -- which is the shape of every sparse number on this
branch: -10% on SplaTAM (small maps, and its forward is the cheap half), null
on MonoGS, null on GSLAM at 780k Gaussians, where renderCUDABackward dominates.

WHAT THIS FILE GUARDS is the claim that lets the fix ship: the early return in
renderCUDABackward is GRADIENT-NEUTRAL. If that is wrong, the failure is the
silent kind this repo keeps meeting -- not a crash, just a tracker converging
to a worse pose, visible only as an ATE regression a fortnight later.

WHAT IT CHECKS
  1. EQUIVALENCE, AGAINST THE KERNEL'S OWN NOISE FLOOR. Same masked forward,
     backward with and without the mask. Not asserted bit-identical: the set of
     atomicAdds is the same either way (masked blocks issue none in any of the
     three inner-loop variants) but their INTERLEAVING is not, and this kernel
     disagrees with itself at ~2e-06 for that reason. So the floor is measured
     first, by running one arm twice, and equivalence is judged against it.
  2. ALL THREE INSTANTIATIONS. renderCUDABackward is a template over
     (COMPUTE_POSE_GRAD, NEED_ALPHA_GRAD) and each combination is a separately
     compiled kernel with a separately compiled early return. The three that
     launch in this repo are SplaTAM (false,false), MonoGS (true,false) and
     Gaussian-SLAM (false,true), and all three are exercised here -- including
     MonoGS's pose gradient, whose block-wide reduction is the one path where
     an early return could plausibly have stranded a barrier.
  3. THE MASK IS ACTUALLY DOING SOMETHING. A masked gradient must differ from a
     DENSE one by orders of magnitude more than the floor. Without this, test 1
     passes trivially if the mask is being dropped somewhere in the plumbing --
     which is the exact bug being fixed, so it is worth a check of its own.
  4. A WRONG-LENGTH MASK RAISES. It used to index out of bounds inside the
     kernel with nothing raising, masking tiles at random offsets. This is the
     hazard utils/pixel_sample.py's tile_dims() exists to avoid on the build
     side; the TORCH_CHECK in rasterize_points.cu closes it on the call side.

NOT A TIMING TEST. A 64x64 toy scene cannot price this and must not be asked
to. This repo's record on synthetic proxies is bad enough to be a rule -- they
missed the real renderCUDABackward share and the real barrier-stall ratio. The
timing number this file prints is a smoke signal, has no pass/fail attached,
and is not a result. The result comes from profiling/legacy/run_sparse_stage_split.sh
on real slam.py.

Run on the Linux GPU VM after `bash rebuild.sh`.
"""
import math

import torch

import diff_gaussian_rasterization as dgr
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    tile_size,
)

torch.manual_seed(0)
device = "cuda"
P = 400
H, W = 64, 64

# Read the grid out of the extension rather than assuming 16x16. The tile size
# is a build variant (DGR_BLOCK_X/Y) and a mask whose length disagrees with the
# grid is exactly what test 4 is about.
BX, BY = tile_size()
TILE_H = (H + BY - 1) // BY
TILE_W = (W + BX - 1) // BX
N_TILES = TILE_H * TILE_W


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
        image_height=H, image_width=W,
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix.transpose(0, 1).contiguous(),
        projmatrix=full_proj.transpose(0, 1).contiguous(),
        projmatrix_raw=proj.transpose(0, 1).contiguous(),
        sh_degree=2,
        campos=torch.tensor([0.0, 0.0, -3.0], device=device),
        prefiltered=False, debug=False,
    )


def make_gaussians():
    means3D = torch.rand(P, 3, device=device) * 2 - 1
    means3D[:, 2] += 3.0
    opacities = torch.sigmoid(torch.randn(P, 1, device=device))
    # Fat on purpose. Each Gaussian must land in SEVERAL tiles, or a masked and
    # an unmasked tile share no Gaussians and the equivalence check never
    # exercises the case it is about: a Gaussian whose gradient comes from a
    # mixture of kept and skipped tiles.
    scales = torch.rand(P, 3, device=device) * 0.08 + 0.05
    raw_rot = torch.randn(P, 4, device=device)
    rotations = raw_rot / raw_rot.norm(dim=-1, keepdim=True)
    sh = torch.randn(P, 9, 3, device=device)
    return means3D, opacities, scales, rotations, sh


CAMERA = make_camera()
BASE = make_gaussians()

# A CHECKERBOARD, not build_tile_mask's top-gradient selection. Deterministic
# (build_tile_mask's uniform half draws from randperm, so two calls disagree),
# and it guarantees every kept tile borders a skipped one, which is where a
# stranded barrier or a half-applied mask would show.
MASK = torch.zeros(N_TILES, dtype=torch.bool, device=device)
MASK.view(TILE_H, TILE_W)[::2, ::2] = True
MASK.view(TILE_H, TILE_W)[1::2, 1::2] = True

# (label, pose_grad, alpha_grad) -- the three template instantiations that
# actually launch in this repo.
ARMS = [
    ("SplaTAM (pose=F, alpha=F)", False, False),
    ("MonoGS  (pose=T, alpha=F)", True, False),
    ("GSLAM   (pose=F, alpha=T)", False, True),
]


def grads(pose_grad, alpha_grad, tile_mask, mask_backward=True):
    """One masked forward + backward. Returns the per-Gaussian gradients.

    mask_backward flips dgr._MASK_BACKWARD, which is the module global the
    wrapper reads per call -- so both arms run THE SAME .so and the standing
    "which arm was which" hazard cannot apply.
    """
    prev = dgr._MASK_BACKWARD
    dgr._MASK_BACKWARD = mask_backward
    try:
        means3D = BASE[0].clone().requires_grad_(True)
        opacities, scales, rotations, sh = [t.clone() for t in BASE[1:]]
        opacities.requires_grad_(True)
        scales.requires_grad_(True)
        means2D = torch.zeros_like(means3D, requires_grad=True)

        theta = rho = None
        if pose_grad:
            theta = torch.zeros(1, 3, device=device, requires_grad=True)
            rho = torch.zeros(1, 3, device=device, requires_grad=True)

        rasterizer = GaussianRasterizer(raster_settings=CAMERA)
        color, radii, depth, alpha, n_touched = rasterizer(
            means3D=means3D, means2D=means2D, opacities=opacities, shs=sh,
            scales=scales, rotations=rotations,
            theta=theta, rho=rho,
            tracking_only=False, alpha_grad=alpha_grad,
            tile_mask=tile_mask,
        )

        # Include every output the arm claims to differentiate, so the loss
        # actually reaches the branch under test. Weighted unevenly: an equal
        # sum lets a sign error in one channel cancel another.
        loss = color.sum() + 0.5 * depth.sum()
        if alpha_grad:
            loss = loss + 0.25 * alpha.sum()
        loss.backward()

        out = [means3D.grad, opacities.grad, scales.grad]
        if pose_grad:
            out += [theta.grad, rho.grad]
        return torch.cat([g.reshape(-1) for g in out])
    finally:
        dgr._MASK_BACKWARD = prev


def _spread(a, b):
    return (a - b).abs().max().item()


def test_equivalence():
    """Masked backward == unmasked backward, judged against the noise floor."""
    print("Test 1: gradient equivalence, all three instantiations")
    ok = True
    for label, pose_grad, alpha_grad in ARMS:
        # The floor FIRST, and from the same arm run twice. atomicAdd ordering
        # is not reproducible, so the question is never "are they equal" but
        # "do they differ by more than this kernel differs from itself".
        a = grads(pose_grad, alpha_grad, MASK, mask_backward=False)
        b = grads(pose_grad, alpha_grad, MASK, mask_backward=False)
        floor = _spread(a, b)

        masked = grads(pose_grad, alpha_grad, MASK, mask_backward=True)
        delta = _spread(masked, a)

        budget = max(10 * floor, 1e-7)
        passed = delta <= budget
        ok = ok and passed
        print(f"  [{'OK  ' if passed else 'FAIL'}] {label}: "
              f"delta {delta:.3e}  floor {floor:.3e}  budget {budget:.3e}")
        if not passed:
            print("         The early return is NOT gradient-neutral. Do not")
            print("         ship this: the symptom downstream is a worse ATE,")
            print("         not an error.")
    return ok


def test_mask_is_live():
    """A masked gradient must differ from a DENSE one, by a lot.

    Test 1 passes trivially if the mask is silently dropped on the way to the
    kernel -- which is the bug being fixed, so it needs its own check.
    """
    print("Test 2: the mask is reaching the kernel at all")
    ok = True
    for label, pose_grad, alpha_grad in ARMS:
        a = grads(pose_grad, alpha_grad, MASK, mask_backward=False)
        b = grads(pose_grad, alpha_grad, MASK, mask_backward=False)
        floor = max(_spread(a, b), 1e-30)

        dense = grads(pose_grad, alpha_grad, None)
        signal = _spread(a, dense)

        passed = signal > max(1000 * floor, 1e-4)
        ok = ok and passed
        print(f"  [{'OK  ' if passed else 'FAIL'}] {label}: masked vs dense "
              f"{signal:.3e} ({signal / floor:.1e}x the floor)")
    return ok


def test_wrong_length_mask_raises():
    """A mask whose length disagrees with the tile grid must not run.

    It used to index out of bounds inside the kernel and mask tiles at random
    offsets, with nothing raising.
    """
    print("Test 3: a wrong-length mask is rejected, not silently misread")
    bad = torch.ones(N_TILES + 3, dtype=torch.bool, device=device)
    try:
        grads(False, False, bad)
    except RuntimeError as exc:
        hit = "tile_mask" in str(exc)
        print(f"  [{'OK  ' if hit else 'FAIL'}] raised: {str(exc).splitlines()[0][:96]}")
        return hit
    print("  [FAIL] a mask of the wrong length ran without complaint")
    return False


def smoke_timing(reps=30):
    """DIAGNOSTIC ONLY -- NO PASS/FAIL, AND NOT A RESULT.

    A 64x64 toy scene is a synthetic proxy, and this repo's proxies have missed
    the real renderCUDABackward share and the real barrier-stall ratio. This
    prints only so a plumbing failure (the mask never reaching the kernel)
    shows up here rather than after an overnight run. Quote nothing from it.
    """
    print("Timing smoke signal -- DIAGNOSTIC, NOT A RESULT")

    def timed(mask_backward):
        for _ in range(5):
            grads(False, False, MASK, mask_backward=mask_backward)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            grads(False, False, MASK, mask_backward=mask_backward)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps

    off = timed(False)
    on = timed(True)
    print(f"  forward-only mask {off:7.3f} ms/iter")
    print(f"  both halves       {on:7.3f} ms/iter   ({100 * (on - off) / off:+.1f}%)")
    print("  (toy scene, whole-iteration wall including Python -- the real")
    print("   split is profiling/legacy/run_sparse_stage_split.sh on slam.py)")


if __name__ == "__main__":
    results = [
        test_equivalence(),
        test_mask_is_live(),
        test_wrong_length_mask_raises(),
    ]
    print()
    smoke_timing()
    print()
    if all(results):
        print("ALL PASS -- masking the backward render changes timing, not gradients.")
    else:
        print("FAILURES -- see above.")
        raise SystemExit(1)
