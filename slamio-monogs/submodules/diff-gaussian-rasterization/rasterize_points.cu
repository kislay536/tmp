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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <c10/cuda/CUDAStream.h>
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
    const torch::Tensor& projmatrix_raw,
    const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& tile_mask,
	const bool compute_n_touched,
	const bool collect_stats,
	const long long binning_capacity,
	const torch::Tensor& binning_overflow)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor n_touched = compute_n_touched ? torch::full({P}, 0, means3D.options().dtype(torch::kInt32)) : torch::full({0}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_opaticy = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor stats = collect_stats ? torch::zeros({2}, means3D.options().dtype(torch::kInt32)) : torch::zeros({0}, means3D.options().dtype(torch::kInt32));

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  int* n_touched_ptr = compute_n_touched ? n_touched.contiguous().data<int>() : nullptr;
	  uint32_t* dev_stats_ptr = collect_stats ? reinterpret_cast<uint32_t*>(stats.contiguous().data<int>()) : nullptr;
	  const bool* tile_mask_ptr = (tile_mask.numel() > 0) ? tile_mask.contiguous().data_ptr<bool>() : nullptr;


	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_depth.contiguous().data<float>(),
		out_opaticy.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		n_touched_ptr,
		tile_mask_ptr,
        debug,
        dev_stats_ptr,
        binning_capacity,
        (binning_overflow.defined() && binning_overflow.numel() > 0)
            ? binning_overflow.contiguous().data_ptr<int>() : nullptr,
        at::cuda::getCurrentCUDAStream());
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_depth, out_opaticy, n_touched, stats);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& projmatrix_raw,
    const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depths,
	const torch::Tensor& dL_dout_opacity,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const bool compute_pose_grad,
	const bool tracking_only,
	const bool skip_color_grad,
	const torch::Tensor& tile_mask)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  // dL_dout_depths is empty when the caller's `depth` output was never
  // consumed (autograd's set_materialize_grads(False) makes an unused
  // output's grad arrive as None, which the Python wrapper turns into an
  // empty tensor here) - e.g. SplaTAM, which packs depth into
  // colors_precomp instead of using the native depth output. In that case
  // skip allocating/zeroing the depth-gradient buffer and skip the
  // depth-gradient bookkeeping in the CUDA kernels entirely.
  const bool need_depth_grad = dL_dout_depths.numel() > 0;

  // Same contract as need_depth_grad, for the rendered ALPHA image. Empty when
  // the caller never consumed the `opacity` output - MonoGS and SplaTAM never
  // do - so their backward is byte-for-byte the path it was before this
  // existed. Gaussian-SLAM DOES consume it: soft_alpha multiplies both
  // tracking residuals by alpha**3, and without this the resulting branch of
  // the pose gradient is silently zero. That alone is worth ~2x on TUM ATE.
  const bool need_alpha_grad = dL_dout_opacity.numel() > 0;

  // HELD IN A NAMED TENSOR, not folded into the ternary below. `t.contiguous()`
  // returns a temporary when t is already non-contiguous, and taking data_ptr
  // off that temporary leaves a pointer into storage freed at the end of the
  // full expression - a dangling device pointer read by a kernel launched
  // afterwards. The masks utils/pixel_sample.py builds are contiguous, so the
  // bug would never fire in the shipped path and would surface only under some
  // future caller that sliced one.
  //
  // The length check is the second half. A mask whose length disagrees with the
  // tile grid indexes out of bounds inside the kernel and NOTHING RAISES - the
  // exact silent failure utils/pixel_sample.py's tile_dims() exists to avoid on
  // the build side. This costs one host-side comparison per backward.
  //
  // .defined() FIRST, and not merely for tidiness: this argument defaults to
  // torch::Tensor(), which is UNDEFINED rather than empty, and numel() on an
  // undefined tensor throws. The Python wrapper always passes a real tensor
  // (empty when dense), so only a C++ caller taking the default would hit it.
  const bool have_mask = tile_mask.defined() && tile_mask.numel() > 0;
  const int grid_tiles = ((W + BLOCK_X - 1) / BLOCK_X) * ((H + BLOCK_Y - 1) / BLOCK_Y);
  TORCH_CHECK(!have_mask || tile_mask.numel() == grid_tiles,
              "tile_mask has ", tile_mask.numel(), " entries but the ", W, "x", H,
              " tile grid has ", grid_tiles, " (", BLOCK_X, "x", BLOCK_Y,
              " tiles). A wrong-length mask masks tiles at random offsets.");
  const torch::Tensor tile_mask_c =
      have_mask ? tile_mask.contiguous() : torch::Tensor();
  const bool* tile_mask_ptr =
      have_mask ? tile_mask_c.data_ptr<bool>() : nullptr;

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_ddepths = need_depth_grad ? torch::zeros({P, 1}, means3D.options()) : torch::zeros({0, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dtau = compute_pose_grad ? torch::zeros({P, 6}, means3D.options()) : torch::zeros({0, 6}, means3D.options());

  if(P != 0)
  {
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H,
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
      projmatrix_raw.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  need_depth_grad ? dL_dout_depths.contiguous().data<float>() : nullptr,
	  need_alpha_grad ? dL_dout_opacity.contiguous().data<float>() : nullptr,
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  need_depth_grad ? dL_ddepths.contiguous().data<float>() : nullptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
      compute_pose_grad ? dL_dtau.contiguous().data<float>() : nullptr,
	  debug,
	  tracking_only,
	  skip_color_grad,
	  need_depth_grad,
	  need_alpha_grad,
	  tile_mask_ptr,
	  at::cuda::getCurrentCUDAStream());
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dtau);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}