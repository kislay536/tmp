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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
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
	const bool compute_n_touched = true,
	const bool collect_stats = false,
	// Capture-safe binning. binning_capacity >= 0 skips the device-to-host
	// readback of num_rendered, making the call recordable into a CUDA graph.
	// binning_overflow is a 1-element int32 CUDA tensor set to 1 on overrun
	// (empty tensor to disable). Defaults reproduce the original path exactly.
	const long long binning_capacity = -1,
	const torch::Tensor& binning_overflow = torch::Tensor());

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
    const torch::Tensor& dL_dout_depth,
    const torch::Tensor& dL_dout_opacity,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const bool compute_pose_grad = false,
	const bool tracking_only = false,
	const bool skip_color_grad = false,
	// The tile mask the MATCHING FORWARD was given, or an empty tensor for a
	// dense backward. Appended last so every existing positional caller is
	// unchanged. Passing an empty tensor here for a call whose forward WAS
	// masked is not an error - it just reproduces the old, slower behaviour,
	// which is exactly what DGR_MASK_BACKWARD=0 asks for.
	const torch::Tensor& tile_mask = torch::Tensor());


torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);