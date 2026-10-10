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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_depth,
			float* out_opacity,
			int* radii = nullptr,
			int* n_touched = nullptr,
			const bool* tile_mask = nullptr,
			bool debug = false,
			uint32_t* dev_stats = nullptr,
			// Capture-safe binning. binning_capacity >= 0 replaces the
			// device-to-host readback of num_rendered with a caller-chosen
			// fixed size, which is what allows this call to be recorded into
			// a CUDA graph. binning_overflow (device int) is set to 1 if the
			// capacity was too small; the host checks it once per frame.
			long long binning_capacity = -1,
			int* binning_overflow = nullptr,
			// Kernels must run on PyTorch's current stream, not the legacy
			// default stream: mixing the two during graph capture fails with
			// "operation would make the legacy stream depend on a capturing
			// blocking stream".
			cudaStream_t stream = 0);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
            const float* projmatrix_raw,
            const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_dpix_depth,
			const float* dL_dpix_opacity,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_ddepths,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dtau,
			bool debug,
			bool tracking_only = false,
			bool skip_color_grad = false,
			bool need_depth_grad = true,
			bool need_alpha_grad = false,
			// Per-tile keep mask for sparse tracking, or nullptr for a dense
			// render. Must be the SAME mask the matching forward() was given -
			// see the early return in renderCUDABackward for why that is a
			// correctness condition and not just a consistency preference.
			const bool* tile_mask = nullptr,
			cudaStream_t stream = 0);
	};
};

#endif