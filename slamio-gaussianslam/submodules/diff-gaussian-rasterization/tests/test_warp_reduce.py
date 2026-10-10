"""Verify the warp-aggregated gradient reduction produces the same gradients.

WHY THIS EXISTS. The failure mode of __shfl_down_sync is not a crash. If a lane
that took an early exit never reaches the shuffle, the result is undefined and
you get SILENTLY WRONG GRADIENTS - a run that completes, reports healthy
counters, and drifts. This branch has produced that exact class of failure
several times already, so the reduction does not get trusted on a timing number.

HOW TO USE. The reduction is a compile-time flag, so this is a two-build test:

    cd submodules/diff-gaussian-rasterization

    # 1. baseline build - TWO runs, to measure how much it disagrees with
    #    ITSELF. The 256-thread atomics are serialised in hardware-determined
    #    order, so this is not deterministic and the disagreement is the
    #    reordering noise floor.
    bash rebuild.sh
    python tests/test_warp_reduce.py --save a.pt
    python tests/test_warp_reduce.py --save b.pt
    python tests/test_warp_reduce.py --noise a.pt b.pt      # prints a tolerance

    # 2. warp build, judged against that measured floor
    bash rebuild.sh warp
    python tests/test_warp_reduce.py --compare a.pt --tol <from step 1>

    # 3. restore whichever build you want to keep
    bash rebuild.sh

    rebuild.sh handles --no-cache-dir and --no-build-isolation and then VERIFIES
    what got installed. A bare `pip install` does neither: pip caches wheels
    keyed on source contents, an env var is not part of that key, and a flagged
    build on unchanged source will silently serve an unflagged cached wheel.

The seed fixes the scene, so both builds rasterize identical Gaussians from an
identical pose and the gradients are compared elementwise.

TWO KERNELS, AND THIS TEST ONLY COVERED ONE. renderCUDABackward is templated on
COMPUTE_POSE_GRAD and the two instantiations reduce in completely different
ways: <C, false> (SplaTAM, Gaussian-SLAM) has every thread atomicAdd directly,
<C, true> (MonoGS) runs a shared-memory tree over the block and writes once from
tid==0. Which one runs is decided by whether `theta` is non-empty. Until
--pose-grad existed this file never passed theta, so the MonoGS path - the one
carrying 67.6% of that kernel's stall time, and the one any barrier or shuffle
work targets next - had NO gradient coverage. Add --pose-grad to every command
above to test it:

    bash rebuild.sh
    python tests/test_warp_reduce.py --pose-grad --save pg_a.pt
    python tests/test_warp_reduce.py --pose-grad --save pg_b.pt
    python tests/test_warp_reduce.py --noise pg_a.pt pg_b.pt
    bash rebuild.sh warp
    python tests/test_warp_reduce.py --pose-grad --compare pg_a.pt --tol <...>

BOTH SIDES OF A --compare MUST AGREE ON --pose-grad. They are different kernels,
not different builds, and comparing across them measures nothing. A mismatch is
detected from the key sets (theta/rho are only saved on the pose-grad path) and
hard-fails rather than quietly skipping those tensors.

THE --noise WORKFLOW DOES TRANSFER TO IT - MEASURED, AGAINST A PREDICTION THAT
IT WOULD NOT. The prediction was that the pose-grad path writes its global
atomics from tid==0 only, one writer per address per block instead of the common
path's 256, so it would agree with itself too closely for --noise to return a
usable floor. Measured on epyc8 in the MonoGS env, n=2, worst over 8 tensors:

    common path    floor ~2.06e-06
    pose-grad      floor  7.05e-08

~30x tighter, NOT "near zero". Cutting the racing writers from 256 to one per
block does not make the path deterministic, because blocks that share a Gaussian
still race at the global atomicAdd. Use --noise here exactly as on the common
path.

The sub-1e-08 warning printed by --noise is kept as a guard for a path that
really is deterministic. The pose-grad path is not that path. frac>1e-3 remains
the discriminator that needs no floor at all, and is what settled the
__syncthreads_count change: 0.000e+00 on all eight tensors, with the worst
rel/scale (6.85e-08) BELOW the same-build floor above.

ON TOLERANCES - AND A CORRECTION. The two paths sum the same values in a
DIFFERENT ORDER, and floating-point addition is not associative, so exact
equality is neither expected nor desirable as a criterion.

The first version of this test used per-element relative error, |a-b|/|a|, and
that was WRONG. Gradients like dL_dmean2D sum signed terms - d.x and d.y are
symmetric about the Gaussian centre - so many elements land near zero through
cancellation, and dividing by such an element turns pure rounding into an
enormous "relative error". The first run duly failed on exactly the four
sign-cancelling tensors (means2D, conic-derived scales/rotations, means3D)
while every sign-definite one (colors, opacities) passed at ~1e-6. That pattern
is a property of the metric, not of the kernel.

Two statistics replace it:

  rel/scale   max|diff| / max|ref|, so near-zero elements cannot dominate.
  frac>1e-3   fraction of elements differing by more than 1e-3 of the tensor
              scale. THIS IS THE DISCRIMINATOR. Summation reordering perturbs
              EVERY element slightly, so the fraction stays near zero. A lane
              dropped from the shuffle destroys the specific Gaussians that
              lane contributed to, so the difference is CONCENTRATED and the
              fraction is visibly non-zero. Diffuse versus concentrated
              separates the two causes; magnitude alone does not.

And the tolerance is MEASURED rather than guessed. The baseline path's 256
per-thread atomics are serialised in hardware-determined order, so it is not
deterministic and disagrees with ITSELF between runs. `--noise` quantifies that
floor from two runs of one build, and the cross-build comparison is judged
against it.
"""
import argparse
import math
import sys

import torch

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except ImportError as e:
    sys.exit(f"diff_gaussian_rasterization not importable: {e}\n"
             "Build it first:  cd submodules/diff-gaussian-rasterization && pip install .")


def build_scene(n=4000, H=240, W=320, seed=0, device="cuda", pose_grad=False):
    """A small deterministic scene. Deliberately dense enough that many
    Gaussians overlap each tile - contention only exists when they do, and a
    sparse scene would test nothing.

    pose_grad SELECTS WHICH KERNEL RUNS. A non-empty `theta` is what sets
    ctx.compute_pose_grad in the autograd Function, which dispatches
    renderCUDABackward<C, true> instead of <C, false> - two different reduction
    schemes in the same source file. Without it this test only ever exercised
    the common path, and the pose-grad branch had no gradient coverage at all.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g, device=device)

    means3D = (r(n, 3) - 0.5) * 2.0
    means3D[:, 2] += 4.0                       # in front of the camera
    means3D.requires_grad_(True)

    scales = (r(n, 3) * 0.05 + 0.01).requires_grad_(True)
    quats = torch.nn.functional.normalize(r(n, 4) - 0.5, dim=1).requires_grad_(True)
    opac = (r(n, 1) * 0.5 + 0.25).requires_grad_(True)
    colors = r(n, 3).requires_grad_(True)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    means2D.retain_grad()

    # MonoGS's cam_rot_delta / cam_trans_delta. Zero is not a placeholder: the
    # deltas are the point the pose Jacobian is linearised about and MonoGS
    # passes zeros every iteration too, so grad_tau does not depend on their
    # value. Shape must be (1, 3) - backward returns grad_tau[3:].view(1, -1)
    # and autograd checks that against the input's shape.
    theta = rho = None
    if pose_grad:
        theta = torch.zeros(1, 3, device=device, requires_grad=True)
        rho = torch.zeros(1, 3, device=device, requires_grad=True)

    fov = 1.0
    znear, zfar = 0.01, 100.0
    tan = math.tan(fov * 0.5)
    proj = torch.zeros(4, 4, device=device)
    proj[0, 0] = 1.0 / tan
    proj[1, 1] = 1.0 / tan
    proj[2, 2] = zfar / (zfar - znear)
    proj[3, 2] = -(zfar * znear) / (zfar - znear)
    proj[2, 3] = 1.0
    view = torch.eye(4, device=device)

    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=tan, tanfovy=tan,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=view, projmatrix=view @ proj, projmatrix_raw=proj,
        sh_degree=0, campos=torch.zeros(3, device=device),
        prefiltered=False, debug=False,
    )
    return settings, dict(means3D=means3D, means2D=means2D, opacities=opac,
                          colors_precomp=colors, scales=scales, rotations=quats,
                          theta=theta, rho=rho)


def run(seed=0, pose_grad=False):
    settings, p = build_scene(seed=seed, pose_grad=pose_grad)
    rasterizer = GaussianRasterizer(raster_settings=settings)
    # theta/rho are None on the common path; forward turns None into an empty
    # tensor, and empty is exactly what leaves compute_pose_grad False.
    out = rasterizer(
        means3D=p["means3D"], means2D=p["means2D"], shs=None,
        colors_precomp=p["colors_precomp"], opacities=p["opacities"],
        scales=p["scales"], rotations=p["rotations"], cov3D_precomp=None,
        theta=p["theta"], rho=p["rho"],
    )
    image = out[0]
    # A weighted loss rather than .sum(): a plain sum can mask sign errors that
    # cancel across the image.
    w = torch.linspace(0.5, 1.5, image.numel(), device=image.device).view_as(image)
    (image * w).sum().backward()

    grads = {
        "means3D": p["means3D"].grad.detach().clone(),
        "means2D": p["means2D"].grad.detach().clone(),
        "opacities": p["opacities"].grad.detach().clone(),
        "colors": p["colors_precomp"].grad.detach().clone(),
        "scales": p["scales"].grad.detach().clone(),
        "rotations": p["rotations"].grad.detach().clone(),
    }
    if pose_grad:
        # The pose gradient itself. It is the whole reason the block-reduction
        # path exists, so a comparison that omits it is not testing that path's
        # purpose - every other tensor here is also produced by the common one.
        grads["theta"] = p["theta"].grad.detach().clone()
        grads["rho"] = p["rho"].grad.detach().clone()
    return grads


def stats(x, y):
    """Scale-relative difference plus how CONCENTRATED it is.

    Per-element relative error is the wrong statistic here and the first
    version of this test used it. Gradients like dL_dmean2D sum signed terms
    (d.x, d.y are symmetric about the Gaussian centre) so many elements sit
    near zero through cancellation. Dividing by such an element magnifies pure
    rounding into a huge "relative error", and indeed the first run failed on
    exactly the sign-cancelling tensors while every sign-definite one passed.

    Two better statistics:
      rel_scale   max|diff| / max|ref|  - difference against the tensor's own
                  scale, so near-zero elements cannot dominate.
      frac_bad    fraction of elements differing by more than 1e-3 of the
                  tensor scale. THIS is the discriminator: reordering perturbs
                  EVERY element a little, while a dropped lane wrecks the few
                  Gaussians that lane contributed to. Diffuse vs concentrated.
    """
    absd = (x - y).abs()
    scale = x.abs().max().clamp_min(1e-20)
    rel_scale = (absd.max() / scale).item()
    frac_bad = (absd > 1e-3 * scale).float().mean().item()
    return rel_scale, frac_bad, scale.item()


def compare(a, b, tol=None, label="A vs B"):
    print(f"\n  {label}")
    print(f"  {'tensor':<12} {'rel/scale':>11} {'frac>1e-3':>11} "
          f"{'scale':>11}   {'verdict' if tol else ''}")
    print("  " + "-" * 62)
    worst, bad = 0.0, []

    # A key mismatch means one side ran with --pose-grad and the other did not,
    # i.e. the two sides executed DIFFERENT KERNELS. Comparing them would
    # produce a plausible-looking table about nothing, which is the failure
    # this suite exists to prevent. Fail on it rather than skipping keys.
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if only_a or only_b:
        bad.append(
            "key mismatch: reference has " + (str(only_a) or "[]") +
            " that this run does not, this run has " + (str(only_b) or "[]") +
            " that the reference does not. One side ran --pose-grad and the "
            "other did not - they are different kernels, not different builds."
        )

    for k in sorted(set(a) & set(b)):
        x, y = a[k].float(), b[k].float().to(a[k].device)
        if x.shape != y.shape:
            bad.append(f"{k}: shape {tuple(x.shape)} vs {tuple(y.shape)}")
            continue
        rs, fb, sc = stats(x, y)
        worst = max(worst, rs)
        verdict = ""
        if tol is not None:
            ok = rs <= tol
            verdict = "ok" if ok else "OVER NOISE"
            if not ok:
                bad.append(f"{k}: rel/scale {rs:.3e} > noise floor {tol:.3e}")
        print(f"  {k:<12} {rs:11.3e} {fb:11.3e} {sc:11.3e}   {verdict}")
    return worst, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", help="write this build's gradients to a file")
    ap.add_argument("--compare", help="compare against a saved file")
    ap.add_argument("--noise", nargs=2, metavar=("A", "B"),
                    help="two files from the SAME build: establishes the "
                         "nondeterminism floor")
    ap.add_argument("--tol", type=float,
                    help="pass threshold on rel/scale; get it from --noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose-grad", action="store_true",
                    help="exercise renderCUDABackward<C, true> - MonoGS's "
                         "block-reduction path - instead of the common path. "
                         "Both sides of a --compare must use the same setting.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a CUDA device")

    # Establish the noise floor from two runs of ONE build. The baseline path
    # has 256 threads atomically adding to each address in whatever order the
    # hardware serialises them, which is NOT deterministic - so two runs of the
    # same binary already differ. That difference is the reordering noise, and
    # it is the only correct yardstick for judging the warp path. Anything else
    # is picking a tolerance out of the air, which is what the first version of
    # this test did.
    if args.noise:
        a = torch.load(args.noise[0], map_location="cuda")
        b = torch.load(args.noise[1], map_location="cuda")
        worst, _ = compare(a, b, tol=None, label="NOISE FLOOR (same build, two runs)")
        print(f"\n  Worst rel/scale from nondeterminism alone: {worst:.3e}")

        # A floor too small to be a tolerance. This was PREDICTED for the
        # pose-grad path and measured false - see the docstring - so the guard
        # stays generic. --noise reads saved files and cannot tell which path
        # produced them, hence the check on magnitude rather than a flag.
        if worst < 1e-8:
            print()
            print("  DO NOT USE THIS AS A TOLERANCE - IT IS TOO SMALL TO MEAN")
            print("  ANYTHING. A floor this low says the build barely disagrees")
            print("  with itself, so judging a reordering against it fails a")
            print("  CORRECT kernel and reports it as broken.")
            print()
            print("  IF YOU ARE ON THE POSE-GRAD PATH, INVESTIGATE RATHER THAN")
            print("  REACHING FOR A TOLERANCE. Its floor was measured at")
            print("  7.05e-08 on epyc8 - only ~30x tighter than the common")
            print("  path's ~2.06e-06, because blocks sharing a Gaussian still")
            print("  race at the global atomicAdd. A floor two orders below")
            print("  that means something else changed.")
            print()
            print("  Otherwise read frac>1e-3, which separates diffuse rounding")
            print("  from a dropped lane with no floor at all, or build one from")
            print("  runs at DIFFERENT --seed values.")
            return 0

        print("  Use this as the tolerance when comparing builds:")
        print(f"      python tests/test_warp_reduce.py --compare {args.noise[0]} "
              f"--tol {worst * 3:.3e}")
        print("  (3x the observed floor, since two samples underestimate the spread)")
        return 0

    grads = run(args.seed, pose_grad=args.pose_grad)
    if args.save:
        torch.save({k: v.cpu() for k, v in grads.items()}, args.save)
        print(f"saved gradients -> {args.save}")
        return 0

    if args.compare:
        ref = torch.load(args.compare, map_location="cuda")
        tol = args.tol
        worst, bad = compare(ref, grads, tol=tol, label="cross-build comparison")
        print()
        if tol is None:
            print("  No tolerance given, so this is descriptive only.")
            print("  Establish the noise floor FIRST - the baseline path's atomics")
            print("  are nondeterministic, so even one build disagrees with itself:")
            print("      python tests/test_warp_reduce.py --save a.pt")
            print("      python tests/test_warp_reduce.py --save b.pt")
            print("      python tests/test_warp_reduce.py --noise a.pt b.pt")
            print()
            print("  READ frac>1e-3 IN THE MEANTIME. Reordering perturbs EVERY")
            print("  element slightly, so that fraction stays near zero. A lane")
            print("  dropped from the shuffle wrecks the specific Gaussians it")
            print("  contributed to, so the difference is CONCENTRATED and the")
            print("  fraction is visibly non-zero. Diffuse vs concentrated is the")
            print("  discriminator, not the magnitude.")
            return 0
        if bad:
            print("  FAIL - differences exceed the measured noise floor:")
            for b_ in bad:
                print(f"    {b_}")
            print()
            print("  Check that every early exit sets the predicate rather than")
            print("  using `continue`, so all 32 lanes reach __shfl_down_sync.")
            return 1
        print(f"  PASS - worst rel/scale {worst:.3e} is within the noise floor.")
        return 0

    print("nothing to do: pass --save, --compare or --noise")
    return 0


if __name__ == "__main__":
    sys.exit(main())
