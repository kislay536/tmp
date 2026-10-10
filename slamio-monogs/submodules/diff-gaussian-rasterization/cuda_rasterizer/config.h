/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB

// ---------------------------------------------------------------------------
// TILE SIZE  (bash rebuild.sh block 8 8   /   DGR_BLOCK_X=8 DGR_BLOCK_Y=8)
//
// 16x16 is 3DGS's original choice and it has NEVER BEEN ABLATED in the
// literature - the paper does not justify it and graphdeco issue #1103 asks
// exactly this question with no published answer. It is worth a sweep here
// because it moves BOTH dominant stalls this branch measured, in opposite
// directions on different models:
//
//   SMALLER TILES (8x8 = 64 threads)
//     + 64 threads race each gradient address instead of 256. SplaTAM and
//       Gaussian-SLAM measured 12.08 and 15.85 RED sectors per request - that
//       is the contention, and this cuts it 4x WITH NO ADDED INSTRUCTIONS.
//       Contrast DGR_WARP_REDUCE, which removed the same contention but added
//       ~50 shuffles per thread and came out 11% SLOWER on SplaTAM.
//     + the block-wide reduction tree is log2(BLOCK_SIZE) deep: 8 steps at 256,
//       6 at 64. MonoGS pays 67.6% of its stalls on barriers.
//     - each Gaussian lands in MORE tiles. Same (2r/T + 1)^2 the tile_footprint
//       probe uses, with T as the variable: at a 8px mean radius, 16 -> 8 is
//       ~2.25x more tile-Gaussian intersections to bin, sort and re-load.
//
//   LARGER TILES (32x16 = 512 threads)
//     the reverse trade, plus fewer blocks/SM.
//
// OCCUPANCY WILL NOT SAVE IT. Shared memory scales with BLOCK_SIZE, so smaller
// tiles free it - but registers and shared memory were measured CO-LIMITING at
// exactly 5 blocks/SM, and this file already records that freeing one alone
// gains nothing.
//
// THIS CHANGES NO ARITHMETIC, only how pixels are grouped into blocks, so the
// gradients are the same up to summation order. Validation is a timing A/B,
// not a gradient check - though the summation order DOES change, so expect
// atomicAdd nondeterminism at the usual ~2e-06 scale, not bit-identity.
//
// CALLERS MUST AGREE. utils/pixel_sample.py builds a per-tile mask whose
// length is the tile grid, so it reads the built value out of the extension
// rather than hardcoding 16. A mismatch is silent and produces a wrong mask.
#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif

// ---------------------------------------------------------------------------
// OCCUPANCY CONTROL FOR THE TWO TILE KERNELS  (bash rebuild.sh minblocks N)
//
// renderCUDA and renderCUDABackward both declare __launch_bounds__(BLOCK_X *
// BLOCK_Y) - the block size ONLY, with no minBlocksPerMultiprocessor. So the
// compiler is told nothing about occupancy and optimises freely for ILP,
// landing at 48 registers/thread. On A100's 65536-register file that is
//
//     48 x 256 = 12288 registers/block  ->  65536/12288 = 5 blocks/SM
//     5 blocks x 8 warps = 40 warps resident, against a maximum of 64
//
// which matches the measured occupancy (45.5% SplaTAM, 43.8% GSLAM, 52.0%
// MonoGS). Registers are the binding constraint: shared memory at 11,264 B
// (common path) would allow ~14 blocks, so freeing shared memory alone cannot
// move this - only registers can.
//
// Supplying the second argument inverts the compiler's problem. It must then
// fit 65536/(N * 256) registers per thread and will spill to local memory to
// comply:
//
//     N=6 -> 42 regs    N=7 -> 36 regs    N=8 -> 32 regs (64 warps, 100%)
//
// THE TRADE IS SPILLS AGAINST RESIDENT WARPS. Both stall profiles this kernel
// shows are LATENCY stalls - barrier (41.8% SplaTAM, 67.6% MonoGS) and shared
// memory (29.1%, 41.2%) - and the standard remedy for those is more warps to
// issue from while one waits. But the inner loop holds the per-pixel colour
// accumulators, transmittance and the dL_dmean2D / dL_dconic / dL_dopacity
// partials live; spilling THOSE would add traffic to the hottest loop and pay
// for nothing.
//
// PRICE IT AT COMPILE TIME BEFORE RUNNING ANYTHING. rebuild.sh passes
// -Xptxas -v, so the build prints per kernel:
//
//     ptxas info: Used 48 registers, 0 bytes spill stores, 0 bytes spill loads
//
// Discard any N whose spill bytes are large in renderCUDABackward without ever
// launching it. Only survivors are worth an A/B.
//
// AND OCCUPANCY IS NOT THROUGHPUT. More warps hide latency; they do not widen
// a saturated pipe. If shared memory is throughput-bound rather than
// latency-bound, 64 warps contend where 40 did. This branch already recorded
// that failure once: SplaTAM's absolute stalls fell 13.80 -> 5.36 under warp
// reduction while the kernel duration ROSE.
//
// DEFAULT IS UNDEFINED = the original single-argument form, bit-identical to
// every measurement in the ladder. This changes no arithmetic in any setting,
// so no gradient verification is required - the whole validation cost is a
// timing A/B.
#ifdef DGR_MIN_BLOCKS_PER_SM
#define TILE_LAUNCH_BOUNDS __launch_bounds__(BLOCK_X * BLOCK_Y, DGR_MIN_BLOCKS_PER_SM)
#else
#define TILE_LAUNCH_BOUNDS __launch_bounds__(BLOCK_X * BLOCK_Y)
#endif

#endif