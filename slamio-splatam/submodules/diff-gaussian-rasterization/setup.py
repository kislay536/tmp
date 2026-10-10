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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

# bash rebuild.sh fastexp     (NOT a bare `pip install` - see rebuild.sh)
#
# Restores the __expf intrinsic at the three Gaussian-falloff sites (see
# DGR_EXP in cuda_rasterizer/auxiliary.h). The DEFAULT is now the precise
# exp(): measured on MonoGS fr1_desk, one variable, __expf was 13.5% worse on
# ATE and 0.46 dB worse on PSNR for no measurable speed gain.
#
# Not re-measured on SplaTAM or Gaussian-SLAM, whose render loads differ - this
# flag exists so that can be tested without a source edit.
_nvcc_args = ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]
if os.environ.get("DGR_FAST_EXP", "") == "1":
    _nvcc_args.append("-DDGR_FAST_EXP")
    print("[setup] DGR_FAST_EXP=1: building with the __expf intrinsic "
          "(lower precision; measured worse on MonoGS)")
else:
    print("[setup] building with precise exp() (default; DGR_FAST_EXP=1 to restore __expf)")

# bash rebuild.sh warp        (NOT a bare `pip install` - see rebuild.sh)
#
# Combines per-Gaussian gradients within each warp (in registers, via
# __shfl_down_sync) before writing, instead of having all 256 threads of a
# block atomically add to the same address.
#
# MEASURED MOTIVATION (profiling/legacy/run_stalls.sh, 30-frame configs):
#
#                 sectors/request   L2 traffic   bandwidth
#   Gaussian-SLAM     15.85           2.99 GB    ~1.37 TB/s   (~30% of L2)
#   SplaTAM           12.08           3.56 GB    ~1.36 TB/s
#   MonoGS             1.50           0.05 GB    ~0.02 TB/s   (block-reduced)
#
# A coalesced 32-bit warp access is 4 sectors/request. The common path sits
# 3-4x above it because 32 lanes reduce to the SAME address and are serialised
# at L2. MonoGS's block reduction already cuts that 67x - it just pays 768
# block.sync() per batch (67.6% of its stalls) to do it. Warp reduction should
# get the traffic win without the barriers.
#
# OFF BY DEFAULT until the gradients are verified against the existing path and
# a real A/B is run. Affects only the common path used by SplaTAM and
# Gaussian-SLAM; MonoGS's pose-gradient path is untouched so far.
if os.environ.get("DGR_WARP_REDUCE", "") == "1":
    _nvcc_args.append("-DDGR_WARP_REDUCE")
    print("[setup] DGR_WARP_REDUCE=1: warp-aggregated gradient reduction "
          "(verify gradients before trusting timings)")

# bash rebuild.sh minblocks N   (NOT a bare `pip install` - see rebuild.sh)
#
# Adds the minBlocksPerMultiprocessor argument to __launch_bounds__ on
# renderCUDA and renderCUDABackward. See TILE_LAUNCH_BOUNDS in
# cuda_rasterizer/config.h for what it does and what it costs.
#
# Short version: those kernels compile to 48 registers/thread, which pins them
# at 5 blocks/SM (40 of A100's 64 warps). Registers are the binding constraint
# - shared memory would allow ~14 blocks. Forcing N blocks caps registers at
# 65536/(N*256) and the compiler spills to comply.
#
# CHANGES NO ARITHMETIC IN ANY SETTING, so unlike DGR_WARP_REDUCE this needs no
# gradient verification - only a timing A/B. Read the spill bytes from the
# -Xptxas -v output below BEFORE spending GPU time on any value.
_min_blocks = os.environ.get("DGR_MIN_BLOCKS_PER_SM", "").strip()
if _min_blocks:
    if not _min_blocks.isdigit() or int(_min_blocks) < 1:
        raise SystemExit(
            f"[setup] DGR_MIN_BLOCKS_PER_SM must be a positive integer, got "
            f"{_min_blocks!r}. Unset it to build the default (unconstrained) form.")
    _nvcc_args.append(f"-DDGR_MIN_BLOCKS_PER_SM={_min_blocks}")
    _regs = 65536 // (int(_min_blocks) * 256)
    print(f"[setup] DGR_MIN_BLOCKS_PER_SM={_min_blocks}: forcing >= {_min_blocks} "
          f"blocks/SM on the tile kernels (caps them at ~{_regs} registers/thread "
          f"on a 65536-register SM; read the spill bytes below)")
else:
    print("[setup] building with unconstrained __launch_bounds__ "
          "(default; DGR_MIN_BLOCKS_PER_SM=N to force occupancy)")

# CEILING PROBE. Replaces every per-Gaussian scatter write in
# renderCUDABackward with a thread-local accumulation, keeping all the gradient
# MATH and removing only the accumulation STRUCTURE. The kernel duration is then
# the floor no reduction rewrite (e.g. porting dL_dtau to SplaTAM/GSLAM) could
# beat. THE GRADIENTS ARE GARBAGE - timing only.
if os.environ.get("DGR_STUB_GRADS", "") == "1":
    if os.environ.get("DGR_WARP_REDUCE", "") == "1":
        raise SystemExit(
            "[setup] DGR_STUB_GRADS cannot be combined with DGR_WARP_REDUCE: "
            "the stub replaces the non-warp-reduce accumulation path only, so "
            "with warp reduction on it would silently measure the UNSTUBBED "
            "kernel and report it as the ceiling. Build the probe without "
            "warp reduction.")
    _nvcc_args.append("-DDGR_STUB_GRADS")
    print("[setup] " + "!" * 66)
    print("[setup] !! DGR_STUB_GRADS=1 - CEILING PROBE BUILD")
    print("[setup] !! renderCUDABackward does NOT accumulate gradients.")
    print("[setup] !! EVERY GRADIENT THIS BUILD PRODUCES IS WRONG.")
    print("[setup] !! Valid for kernel TIMING only. Any ATE, PSNR or loss from")
    print("[setup] !! this build is meaningless. Rebuild before any real run.")
    print("[setup] " + "!" * 66)

# TILE SIZE. 16x16 is 3DGS's original and unablated choice; see the long note
# in cuda_rasterizer/config.h for why it is worth sweeping and what it trades.
# The DEFINE is BLOCK_X/BLOCK_Y (config.h now #ifndef-guards them) while the
# ENV is DGR_-prefixed like every other flag here.
#
# CONSTRAINTS, checked here so a bad value fails at build time rather than as a
# launch failure 40 minutes into a run:
#   - BLOCK_X * BLOCK_Y <= 1024, the CUDA maximum threads per block, and the
#     tile kernels declare __launch_bounds__(BLOCK_X * BLOCK_Y)
#   - a multiple of 32 keeps whole warps, otherwise every block wastes a
#     partial warp and the warp-level primitives in the reduce path degrade
_block_x = os.environ.get("DGR_BLOCK_X", "").strip()
_block_y = os.environ.get("DGR_BLOCK_Y", "").strip()
if _block_x or _block_y:
    if not (_block_x.isdigit() and _block_y.isdigit()):
        raise SystemExit(
            f"[setup] DGR_BLOCK_X and DGR_BLOCK_Y must BOTH be set to positive "
            f"integers, got {_block_x!r} and {_block_y!r}. Unset both to build "
            f"the 16x16 default.")
    _bx, _by = int(_block_x), int(_block_y)
    if _bx < 1 or _by < 1:
        raise SystemExit(f"[setup] tile dims must be positive, got {_bx}x{_by}")
    if _bx * _by > 1024:
        raise SystemExit(
            f"[setup] {_bx}x{_by} = {_bx * _by} threads/block exceeds the CUDA "
            f"maximum of 1024.")
    if (_bx * _by) % 32 != 0:
        raise SystemExit(
            f"[setup] {_bx}x{_by} = {_bx * _by} threads/block is not a multiple "
            f"of 32, so every block would waste a partial warp. Pick dims whose "
            f"product is a multiple of 32.")
    _nvcc_args.append(f"-DBLOCK_X={_bx}")
    _nvcc_args.append(f"-DBLOCK_Y={_by}")
    print(f"[setup] DGR_BLOCK_X/Y: building {_bx}x{_by} tiles "
          f"({_bx * _by} threads/block, {_bx * _by // 32} warps). "
          f"utils/pixel_sample.py reads this back from the extension - do not "
          f"hardcode it anywhere.")
else:
    print("[setup] building with 16x16 tiles (default; DGR_BLOCK_X/DGR_BLOCK_Y "
          "to sweep)")

# -Xptxas -v makes ptxas print, per kernel, the register count and the spill
# store/load bytes. That is what prices a DGR_MIN_BLOCKS_PER_SM setting WITHOUT
# RUNNING ANYTHING: a value that spills heavily in renderCUDABackward can be
# discarded from the build log alone. Always on - it costs nothing and it is
# the only place the register count is visible.
_nvcc_args += ["-Xptxas", "-v"]

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            # BOTH keys. "nvcc" applies only to .cu files; ext.cpp is compiled
            # by the C++ compiler under "cxx" and never saw these defines.
            #
            # That made the build-variant introspection in ext.cpp report
            # warp_reduce=False on EVERY build, including builds where the
            # kernel genuinely had it - the .cu files got the flag, the .cpp
            # did not. A diagnostic added to remove doubt was itself lying,
            # which cost more than the doubt did.
            #
            # The -I glm include is nvcc-only; only the -D flags need to reach
            # both compilers.
            extra_compile_args={
                "nvcc": _nvcc_args,
                "cxx": [a for a in _nvcc_args if a.startswith("-D")],
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
