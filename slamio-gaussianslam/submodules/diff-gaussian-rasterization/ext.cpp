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

#include <torch/extension.h>
#include "rasterize_points.h"
// For BLOCK_X / BLOCK_Y below. NOT pulled in by rasterize_points.h, and
// without it dgr_block_x() fails to compile rather than silently reporting a
// default - which is the outcome we want, given the last introspection helper
// added to remove doubt reported the wrong value on every build.
#include "cuda_rasterizer/config.h"

// Report which compile-time variants this .so was built with.
//
// The build prints its flags, but a RUN did not - so a measurement could be
// attributed to the wrong binary simply by losing track of which build was
// installed. That happened: a SplaTAM A/B came out 11.3% apart and the first
// question was "which arm was which". A run should state what it is.
static bool dgr_warp_reduce_enabled() {
#ifdef DGR_WARP_REDUCE
  return true;
#else
  return false;
#endif
}
static bool dgr_fast_exp_enabled() {
#ifdef DGR_FAST_EXP
  return true;
#else
  return false;
#endif
}
// 0 means the default, unconstrained __launch_bounds__ (block size only).
// Any other value is the minBlocksPerMultiprocessor the tile kernels were
// compiled to hit. Reported per RUN because the two arms of this A/B differ
// ONLY in a compile-time constant - there is nothing in a config, a log or a
// command line to tell them apart afterwards.
static int dgr_min_blocks_per_sm() {
#ifdef DGR_MIN_BLOCKS_PER_SM
  return DGR_MIN_BLOCKS_PER_SM;
#else
  return 0;
#endif
}

// The tile dimensions this .so was compiled with. Reported per RUN for the
// same reason as min_blocks_per_sm - the arms of a tile-size A/B differ only
// in a compile-time constant. It is also load-bearing rather than merely
// diagnostic: utils/pixel_sample.py sizes its per-tile mask from this, and a
// stale hardcoded 16 against an 8x8 build produces a wrong-length mask with no
// error.
static int dgr_block_x() { return BLOCK_X; }
static int dgr_block_y() { return BLOCK_Y; }

// True when this .so was built as a CEILING PROBE, whose gradients are garbage
// by construction. Reported per run and shouted about at import, because the
// failure mode is silent: a stub build runs, converges to nonsense, and
// produces an ATE that looks like a number.
static bool dgr_stub_grads() {
#ifdef DGR_STUB_GRADS
  return true;
#else
  return false;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("warp_reduce_enabled", &dgr_warp_reduce_enabled);
  m.def("fast_exp_enabled", &dgr_fast_exp_enabled);
  m.def("min_blocks_per_sm", &dgr_min_blocks_per_sm);
  m.def("block_x", &dgr_block_x);
  m.def("block_y", &dgr_block_y);
  m.def("stub_grads", &dgr_stub_grads);
}