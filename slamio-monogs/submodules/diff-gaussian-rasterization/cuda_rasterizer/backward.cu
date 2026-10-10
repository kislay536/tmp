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

#include "backward.h"
#include "auxiliary.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Sum one value across the 32 lanes of a warp, in registers.
//
// Threads in a warp execute in lockstep, so they are ALREADY synchronised -
// this needs no __syncthreads() and touches no shared memory. That is the
// entire point: the block-wide reduction in the pose-gradient path achieves the
// same traffic saving but pays ~10 block-wide barriers PER GAUSSIAN for it:
// 1 for the skip vote (was 3, before __syncthreads_count collapsed them) plus
// 9 for the shared-memory tree, i.e. 2561 per 256-Gaussian batch.
//
// Only lane 0 holds the full sum afterwards; the other lanes hold partial
// sums and must not be used.
//
// EVERY LANE IN THE MASK MUST REACH THIS. __shfl_down_sync with a lane missing
// is undefined behaviour, which here means silently wrong gradients rather than
// a crash - hence the predicated (rather than `continue`-based) loop at the
// call site.
__device__ __forceinline__ float warpReduceSum(float v)
{
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1)
		v += __shfl_down_sync(0xffffffffu, v, offset);
	return v;
}

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs,  float *dL_dtau, bool tracking_only)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to.
	// During tracking_only, the coefficient gradients themselves are never
	// consumed (SH coefficients are not pose-dependent) - only the
	// direction-derivative terms below (dRGBdx/dRGBdy/dRGBdz) feed dL_dmean/dL_dtau,
	// so their computation is kept but the dL_dsh stores are skipped.
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	if (!tracking_only) dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		if (!tracking_only) {
			dL_dsh[1] = dRGBdsh1 * dL_dRGB;
			dL_dsh[2] = dRGBdsh2 * dL_dRGB;
			dL_dsh[3] = dRGBdsh3 * dL_dRGB;
		}

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			if (!tracking_only) {
				dL_dsh[4] = dRGBdsh4 * dL_dRGB;
				dL_dsh[5] = dRGBdsh5 * dL_dRGB;
				dL_dsh[6] = dRGBdsh6 * dL_dRGB;
				dL_dsh[7] = dRGBdsh7 * dL_dRGB;
				dL_dsh[8] = dRGBdsh8 * dL_dRGB;
			}

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				if (!tracking_only) {
					dL_dsh[9] = dRGBdsh9 * dL_dRGB;
					dL_dsh[10] = dRGBdsh10 * dL_dRGB;
					dL_dsh[11] = dRGBdsh11 * dL_dRGB;
					dL_dsh[12] = dRGBdsh12 * dL_dRGB;
					dL_dsh[13] = dRGBdsh13 * dL_dRGB;
					dL_dsh[14] = dRGBdsh14 * dL_dRGB;
					dL_dsh[15] = dRGBdsh15 * dL_dRGB;
				}

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);

	if (dL_dtau != nullptr) {
		dL_dtau[6 * idx + 0] += -dL_dmean.x;
		dL_dtau[6 * idx + 1] += -dL_dmean.y;
		dL_dtau[6 * idx + 2] += -dL_dmean.z;
	}

}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov,
	float *dL_dtau)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;

	if (dL_dtau != nullptr) {
		SE3 T_CW(view_matrix);
		mat33 R = T_CW.R().data();
		mat33 dpC_drho = mat33::identity();
		mat33 dpC_dtheta = -mat33::skew_symmetric(t);
		float dL_dt[6];
		for (int i = 0; i < 3; i++) {
			float3 c_rho = dpC_drho.cols[i];
			float3 c_theta = dpC_dtheta.cols[i];
			dL_dt[i] = dL_dtx * c_rho.x + dL_dty * c_rho.y + dL_dtz * c_rho.z;
			dL_dt[i + 3] = dL_dtx * c_theta.x + dL_dty * c_theta.y + dL_dtz * c_theta.z;
		}
		for (int i = 0; i < 6; i++) {
			dL_dtau[6 * idx + i] += dL_dt[i];
		}

		float dL_dW00 = J[0][0] * dL_dT00;
		float dL_dW01 = J[0][0] * dL_dT01;
		float dL_dW02 = J[0][0] * dL_dT02;
		float dL_dW10 = J[1][1] * dL_dT10;
		float dL_dW11 = J[1][1] * dL_dT11;
		float dL_dW12 = J[1][1] * dL_dT12;
		float dL_dW20 = J[0][2] * dL_dT00 + J[1][2] * dL_dT10;
		float dL_dW21 = J[0][2] * dL_dT01 + J[1][2] * dL_dT11;
		float dL_dW22 = J[0][2] * dL_dT02 + J[1][2] * dL_dT12;

		float3 c1 = R.cols[0];
		float3 c2 = R.cols[1];
		float3 c3 = R.cols[2];

		float dL_dW_data[9];
		dL_dW_data[0] = dL_dW00;
		dL_dW_data[3] = dL_dW01;
		dL_dW_data[6] = dL_dW02;
		dL_dW_data[1] = dL_dW10;
		dL_dW_data[4] = dL_dW11;
		dL_dW_data[7] = dL_dW12;
		dL_dW_data[2] = dL_dW20;
		dL_dW_data[5] = dL_dW21;
		dL_dW_data[8] = dL_dW22;

		mat33 dL_dW(dL_dW_data);
		float3 dL_dWc1 = dL_dW.cols[0];
		float3 dL_dWc2 = dL_dW.cols[1];
		float3 dL_dWc3 = dL_dW.cols[2];

		mat33 n_W1_x = -mat33::skew_symmetric(c1);
		mat33 n_W2_x = -mat33::skew_symmetric(c2);
		mat33 n_W3_x = -mat33::skew_symmetric(c3);

		float3 dL_dtheta = {};
		dL_dtheta.x = dot(dL_dWc1, n_W1_x.cols[0]) + dot(dL_dWc2, n_W2_x.cols[0]) +
					dot(dL_dWc3, n_W3_x.cols[0]);
		dL_dtheta.y = dot(dL_dWc1, n_W1_x.cols[1]) + dot(dL_dWc2, n_W2_x.cols[1]) +
					dot(dL_dWc3, n_W3_x.cols[1]);
		dL_dtheta.z = dot(dL_dWc1, n_W1_x.cols[2]) + dot(dL_dWc2, n_W2_x.cols[2]) +
					dot(dL_dWc3, n_W3_x.cols[2]);

		dL_dtau[6 * idx + 3] += dL_dtheta.x;
		dL_dtau[6 * idx + 4] += dL_dtheta.y;
		dL_dtau[6 * idx + 5] += dL_dtheta.z;
	} // end if (dL_dtau != nullptr)


}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float *viewmatrix,
	const float* proj,
	const float *proj_raw,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float *dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float *dL_dtau,
	bool tracking_only,
	bool need_depth_grad)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient update due to computing depths. Skipped entirely when
	// the caller never consumes the native depth output (dL_ddepth would be
	// nullptr in that case) - e.g. SplaTAM, which packs depth into
	// colors_precomp instead and never needs this term.
	float dL_dpCz = 0.0f;
	if (need_depth_grad) {
		dL_dpCz = dL_ddepth[idx];
		dL_dmeans[idx].x += dL_dpCz * viewmatrix[2];
		dL_dmeans[idx].y += dL_dpCz * viewmatrix[6];
		dL_dmeans[idx].z += dL_dpCz * viewmatrix[10];
	}

	if (dL_dtau != nullptr) {
		float alpha = 1.0f * m_w;
		float beta = -m_hom.x * m_w * m_w;
		float gamma = -m_hom.y * m_w * m_w;

		float a = proj_raw[0];
		float b = proj_raw[5];
		float e = proj_raw[11];

		SE3 T_CW(viewmatrix);
		float3 p_C = T_CW * m;
		mat33 dp_C_d_rho = mat33::identity();
		mat33 dp_C_d_theta = -mat33::skew_symmetric(p_C);

		float3 d_proj_dp_C1 = make_float3(alpha * a, 0.f, beta * e);
		float3 d_proj_dp_C2 = make_float3(0.f, alpha * b, gamma * e);

		float3 d_proj_dp_C1_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C1;
		float3 d_proj_dp_C2_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C2;
		float3 d_proj_dp_C1_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C1;
		float3 d_proj_dp_C2_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C2;

		float2 dmean2D_dtau[6];
		dmean2D_dtau[0].x = d_proj_dp_C1_d_rho.x;
		dmean2D_dtau[1].x = d_proj_dp_C1_d_rho.y;
		dmean2D_dtau[2].x = d_proj_dp_C1_d_rho.z;
		dmean2D_dtau[3].x = d_proj_dp_C1_d_theta.x;
		dmean2D_dtau[4].x = d_proj_dp_C1_d_theta.y;
		dmean2D_dtau[5].x = d_proj_dp_C1_d_theta.z;

		dmean2D_dtau[0].y = d_proj_dp_C2_d_rho.x;
		dmean2D_dtau[1].y = d_proj_dp_C2_d_rho.y;
		dmean2D_dtau[2].y = d_proj_dp_C2_d_rho.z;
		dmean2D_dtau[3].y = d_proj_dp_C2_d_theta.x;
		dmean2D_dtau[4].y = d_proj_dp_C2_d_theta.y;
		dmean2D_dtau[5].y = d_proj_dp_C2_d_theta.z;

		float dL_dt[6];
		for (int i = 0; i < 6; i++) {
			dL_dt[i] = dL_dmean2D[idx].x * dmean2D_dtau[i].x + dL_dmean2D[idx].y * dmean2D_dtau[i].y;
		}
		for (int i = 0; i < 6; i++) {
			dL_dtau[6 * idx + i] += dL_dt[i];
		}

		for (int i = 0; i < 3; i++) {
			float3 c_rho = dp_C_d_rho.cols[i];
			float3 c_theta = dp_C_d_theta.cols[i];
			dL_dtau[6 * idx + i] += dL_dpCz * c_rho.z;
			dL_dtau[6 * idx + i + 3] += dL_dpCz * c_theta.z;
		}
	} // end if (dL_dtau != nullptr)



	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh, dL_dtau, tracking_only);

	// Compute gradient updates due to computing covariance from scale/rotation.
	// Skippable when tracking_only and dL_dtau is active: dL_dtau's rotation
	// term already supersedes dL_dscale/dL_drot for the dL_dtau pose-gradient
	// mechanism. NOT skippable when dL_dtau is nullptr (e.g. SplaTAM's
	// transform_to_frame reparam path), where dL_drot can itself be the
	// pose-gradient carrier for anisotropic Gaussians.
	if (scales && !(tracking_only && dL_dtau != nullptr))
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

template <typename T>
__device__ void inline reduce_helper(int lane, int i, T *data) {
  if (lane < i) {
    data[lane] += data[lane + i];
  }
}

template <typename group_t, typename... Lists>
__device__ void render_cuda_reduce_sum(group_t g, Lists... lists) {
  int lane = g.thread_rank();
  g.sync();

  for (int i = g.size() / 2; i > 0; i /= 2) {
    (...,
     reduce_helper(
         lane, i, lists)); // Fold expression: apply reduce_helper for each list
    g.sync();
  }
}


// Backward version of the rendering procedure.
// COMPUTE_POSE_GRAD=true  (MonoGS): block-level reduction, __syncthreads_count voting
// COMPUTE_POSE_GRAD=false (SplaTAM/GSLAM): one sync per batch, direct per-thread atomicAdds
//
// BARRIER BUDGET of the pose-grad path, per Gaussian per block:
//   1  __syncthreads_count (the skip vote; was 3 separate block.sync())
//   1  render_cuda_reduce_sum's leading g.sync()
//   8  its shared-memory tree, one g.sync() per halving step
//  ---
//  10  was 12
//
// The tree's 9 are the remaining target and they do NOT come out with a
// barrier trick - they come out by replacing the shared-memory tree with
// __shfl_down_sync, which is a separate change. Do not bundle the two: this
// one is arithmetically identical to its predecessor and the shuffle is not.
// NEED_ALPHA_GRAD IS A TEMPLATE PARAMETER, NOT A RUNTIME BOOL, and that is
// deliberate. Only Gaussian-SLAM asks for the rendered-alpha gradient; MonoGS
// and SplaTAM must be able to state that this costs them NOTHING, not "a
// predictable branch". As a template argument the whole block below - and the
// two floats it carries in registers - is gone at compile time for them, so
// their instantiation is the kernel they had before this existed. Register
// pressure in this kernel is priced explicitly by rebuild.sh's ptxas summary,
// which is why "probably free" was not good enough.
template <uint32_t C, bool COMPUTE_POSE_GRAD, bool NEED_ALPHA_GRAD>
__global__ void TILE_LAUNCH_BOUNDS
renderCUDABackward(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixels_depth,
	// GAUSSIAN-SLAM'S SOFT-ALPHA GRADIENT. The rendered alpha image
	// (out_opacity = 1 - T) is a differentiable OUTPUT, not just a mask:
	// Gaussian-SLAM's tracking loss multiplies both the colour and the depth
	// residual by alpha**3 (soft_alpha, on by default in every shipped GSLAM
	// config). Dropping this input silently zeroes that whole branch of the
	// pose gradient - the tracker still converges, just to a worse pose, so
	// the failure shows up ONLY as a degraded ATE, never as an error.
	// nullptr / need_alpha_grad false when the caller never consumed the
	// output, which is the case for MonoGS and SplaTAM, so their cost is
	// unchanged.
	const float* __restrict__ dL_dpixels_opacity,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths,
	bool tracking_only,
	bool skip_color_grad,
	bool need_depth_grad,
	// SPARSE TILE SAMPLING, BACKWARD HALF. Null for every dense render, in
	// which case this kernel is byte-for-byte the one that existed before.
	const bool* __restrict__ tile_mask)
{
	auto block = cg::this_thread_block();
	auto tid = block.thread_rank();

	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

	// MASKED TILES RETURN HERE, AND THIS IS THE HALF OF SPARSE THAT WAS NEVER
	// WIRED UP.
	//
	// forward.cu has taken this same early return since tile masking landed,
	// under a comment asserting the backward needed no mask of its own: a
	// masked tile gets n_contrib = 0 written for every one of its pixels, so
	// `last_contributor` is 0 here, so `skip` / `active` is false for every
	// Gaussian in all three inner-loop variants below, and every write this
	// kernel performs is gated on that predicate. THE ARITHMETIC PART OF THAT
	// CLAIM IS TRUE - which is why this return changes no gradient, and why
	// the check below is a timing A/B and not a numerical one.
	//
	// What it missed is that the predicate gates the MATH, not the LOOP. With
	// n_contrib = 0 a masked tile still ran, per round of its full Gaussian
	// range: the collective global->shared load of point_list, means2D,
	// conic_opacity, colors[C] and depths, one block.sync(), and then
	// BLOCK_SIZE trips through the inner loop each paying a
	// __syncthreads_count (pose-grad path) or a __ballot_sync (fast path) to
	// discover there was nothing to do. Roughly 9 floats x 256 threads of
	// memory traffic and one barrier per round, for a tile that writes
	// nothing.
	//
	// That cost is proportional to the tile's RANGE LENGTH, which grows as the
	// map fills - so sparse was buying forward-render time only, against a
	// backward bill that grew underneath it. It is the shape of the numbers
	// this was measured against: -10% on SplaTAM (small maps, render-heavy,
	// and its forward is the cheap half), and null on MonoGS and on GSLAM at
	// 780k Gaussians, where renderCUDABackward is the dominant kernel and the
	// mask was doing precisely nothing to it.
	//
	// STILL NOT SHRUNK BY THE MASK, deliberately, and this is the boundary of
	// the claim: preprocessCUDA is per-Gaussian, and duplicateWithKeys /
	// RadixSort / identifyTileRanges bin and sort every tile-Gaussian
	// intersection including those of masked tiles. Masking at key emission
	// would shrink those too; it also changes num_rendered, which the binning
	// capacity path sizes buffers from, so it is a separate change.
	if (tile_mask != nullptr && !tile_mask[block.group_index().y * horizontal_blocks + block.group_index().x])
		return;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C] = { 0 };
	float accum_rec_depth = 0;
	float dL_dpixel_depth = 0;
	// Suffix-accumulated alpha - the exact analogue of accum_rec / accum_rec_depth
	// for the out_opacity channel. See the accumulation sites below.
	// Dead for NEED_ALPHA_GRAD == false: every use below is behind
	// `if constexpr`, so the branches are DISCARDED at compile time (not merely
	// optimised away) and these two carry no registers in that instantiation.
	float accum_rea = 0;
	float dL_dpixel_alpha = 0;
	if (inside) {
		#pragma unroll
		for (int i = 0; i < C; i++) {
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		}
		if (need_depth_grad) {
			dL_dpixel_depth = dL_dpixels_depth[pix_id];
		}
		if constexpr (NEED_ALPHA_GRAD) {
			dL_dpixel_alpha = dL_dpixels_opacity[pix_id];
		}
	}

	float last_alpha = 0.f;
	float last_color[C] = { 0.f };
	float last_depth = 0.f;

	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;

#ifdef DGR_STUB_GRADS
	// CEILING PROBE. See the DGR_STUB_GRADS block further down. This is the
	// thread-local sink that replaces every per-Gaussian scatter write, so the
	// gradient MATH still runs and only the ACCUMULATION STRUCTURE is removed.
	float stub_sink = 0.f;
#endif

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory (back-to-front order).
		const int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[tid] = coll_id;
			collected_xy[tid] = points_xy_image[coll_id];
			collected_conic_opacity[tid] = conic_opacity[coll_id];
			#pragma unroll
			for (int ii = 0; ii < C; ii++) {
				collected_colors[ii * BLOCK_SIZE + tid] = colors[coll_id * C + ii];
			}
			if (need_depth_grad) {
				collected_depths[tid] = depths[coll_id];
			}
		}

		if constexpr (COMPUTE_POSE_GRAD) {
			// MonoGS path: block-level reduction + an all-skip vote, so a batch
			// slot no pixel in this block contributes to costs no reduction.
			// Shared reduction arrays are only allocated in this instantiation.
			__shared__ float2 dL_dmean2D_shared[BLOCK_SIZE];
			__shared__ float3 dL_dcolors_shared[BLOCK_SIZE];
			__shared__ float  dL_ddepths_shared[BLOCK_SIZE];
			__shared__ float  dL_dopacity_shared[BLOCK_SIZE];
			__shared__ float4 dL_dconic2D_shared[BLOCK_SIZE];
			// NO skip_counter: __syncthreads_count(pred) IS a block-wide count
			// of a predicate WITH barrier semantics, in a single instruction.
			// See the vote below for what it replaced.

			// Publish this round's collected_* writes ONCE per batch instead of
			// once per Gaussian. The fast path below already does exactly this;
			// the pose-grad path paid it up to 256 times per round because its
			// per-Gaussian barrier doubled as the publish.
			block.sync();

			for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
				bool skip = done;
				contributor = done ? contributor : contributor - 1;
				skip |= contributor >= last_contributor;

				const float2 xy = collected_xy[j];
				const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
				const float4 con_o = collected_conic_opacity[j];
				const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
				skip |= power > 0.0f;

				const float G = DGR_EXP(power);
				const float alpha = min(0.99f, con_o.w * G);
				skip |= alpha < 1.0f / 255.0f;

				// ONE barrier replacing THREE, doing both jobs at once:
				//
				//   (a) counts `skip` across the block - exactly what the
				//       reset / atomicAdd / sync trio computed, minus a
				//       shared int and a shared-memory atomic.
				//   (b) separates the PREVIOUS iteration's tid==0 read of
				//       dL_*_shared[0] from THIS iteration's writes to
				//       dL_*_shared[tid] below. That was the job of the
				//       barrier at the top of the loop.
				//
				// The collected_*[j] reads above need no barrier of their own:
				// they are published once before the loop and nothing in the
				// loop writes them. Every thread also passes this barrier
				// AFTER its reads - including on the all-skip `continue` path -
				// so the next round's collected_* writes cannot race them.
				//
				// The count is BLOCK-UNIFORM, so every thread takes the same
				// `continue`. A divergent __syncthreads_count would hang.
				if (__syncthreads_count(skip) == BLOCK_SIZE) {
					continue;
				}

				T = skip ? T : T / (1.f - alpha);
				const float dchannel_dcolor = alpha * T;

				float dL_dalpha = 0.0f;
				const int global_id = collected_id[j];
				float local_dL_dcolors[3];
				#pragma unroll
				for (int ch = 0; ch < C; ch++)
				{
					const float c = collected_colors[ch * BLOCK_SIZE + j];
					accum_rec[ch] = skip ? accum_rec[ch] : last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
					last_color[ch] = skip ? last_color[ch] : c;
					const float dL_dchannel = dL_dpixel[ch];
					dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
					local_dL_dcolors[ch] = skip ? 0.0f : dchannel_dcolor * dL_dchannel;
				}
				dL_dcolors_shared[tid].x = local_dL_dcolors[0];
				dL_dcolors_shared[tid].y = local_dL_dcolors[1];
				dL_dcolors_shared[tid].z = local_dL_dcolors[2];

				if (need_depth_grad) {
					const float depth = collected_depths[j];
					accum_rec_depth = skip ? accum_rec_depth : last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
					last_depth = skip ? last_depth : depth;
					dL_dalpha += (depth - accum_rec_depth) * dL_dpixel_depth;
					dL_ddepths_shared[tid] = skip ? 0.f : dchannel_dcolor * dL_dpixel_depth;
				} else {
					dL_ddepths_shared[tid] = 0.f;
				}

				// d(out_opacity)/d(alpha_i). out_opacity = 1 - T_final, so the
				// derivative w.r.t. THIS Gaussian's alpha is prod_{j>i}(1 - alpha_j),
				// and (1 - accum_rea) is exactly that product once accum_rea has been
				// advanced with the PREVIOUS last_alpha. Hence the placement: above
				// `last_alpha = alpha`, and above the `*= T` that supplies the
				// remaining T_i factor. Unlike colour and depth there is no
				// per-Gaussian payload to scatter - alpha has no analogue of
				// dL_dcolors / dL_ddepths, only this contribution to dL_dalpha - so
				// this adds no atomics and no shared-memory reduction.
				if constexpr (NEED_ALPHA_GRAD) {
					accum_rea = skip ? accum_rea : last_alpha + (1.f - last_alpha) * accum_rea;
					dL_dalpha += (1.f - accum_rea) * dL_dpixel_alpha;
				}

				dL_dalpha *= T;
				last_alpha = skip ? last_alpha : alpha;

				float bg_dot_dpixel = 0.f;
				#pragma unroll
				for (int ii = 0; ii < C; ii++) {
					bg_dot_dpixel += bg_color[ii] * dL_dpixel[ii];
				}
				dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

				const float dL_dG = con_o.w * dL_dalpha;
				const float gdx = G * d.x;
				const float gdy = G * d.y;
				const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
				const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

				dL_dmean2D_shared[tid].x = skip ? 0.f : dL_dG * dG_ddelx * ddelx_dx;
				dL_dmean2D_shared[tid].y = skip ? 0.f : dL_dG * dG_ddely * ddely_dy;
				dL_dconic2D_shared[tid].x = skip ? 0.f : -0.5f * gdx * d.x * dL_dG;
				dL_dconic2D_shared[tid].y = skip ? 0.f : -0.5f * gdx * d.y * dL_dG;
				dL_dconic2D_shared[tid].w = skip ? 0.f : -0.5f * gdy * d.y * dL_dG;
				dL_dopacity_shared[tid] = skip ? 0.f : G * dL_dalpha;

				render_cuda_reduce_sum(block,
					dL_dmean2D_shared,
					dL_dconic2D_shared,
					dL_dopacity_shared,
					dL_dcolors_shared,
					dL_ddepths_shared
				);

				if (tid == 0) {
					atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2D_shared[0].x);
					atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2D_shared[0].y);
					atomicAdd(&dL_dconic2D[global_id].x, dL_dconic2D_shared[0].x);
					atomicAdd(&dL_dconic2D[global_id].y, dL_dconic2D_shared[0].y);
					atomicAdd(&dL_dconic2D[global_id].w, dL_dconic2D_shared[0].w);
					if (!tracking_only)
						atomicAdd(&dL_dopacity[global_id],   dL_dopacity_shared[0]);
					if (!skip_color_grad) {
						atomicAdd(&dL_dcolors[global_id * C + 0], dL_dcolors_shared[0].x);
						atomicAdd(&dL_dcolors[global_id * C + 1], dL_dcolors_shared[0].y);
						atomicAdd(&dL_dcolors[global_id * C + 2], dL_dcolors_shared[0].z);
					}
					if (need_depth_grad)
						atomicAdd(&dL_ddepths[global_id],    dL_ddepths_shared[0]);
				}
			}
		} else {
			// Fast path (SplaTAM / GSLAM): one sync per batch for shared-memory
			// coherence, then each thread writes its own atomicAdds directly.
			// Eliminates ~8 block.sync() calls per Gaussian vs the MonoGS path.
			block.sync();
			for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
				// PREDICATED, NOT `continue`. Every thread must reach the
				// reduction below: __shfl_down_sync requires all 32 lanes named
				// in the mask to participate, and a lane that took an early exit
				// would never arrive. Missing lanes give UNDEFINED results -
				// silently wrong gradients, not a crash.
				//
				// Both builds share this restructure so the A/B isolates the
				// REDUCTION rather than the rewrite. With DGR_WARP_REDUCE off
				// the arithmetic and the writes are identical to the original;
				// only the shape of the control flow changed.
				bool active = inside;
				if (active) {
					contributor--;
					active = (contributor < last_contributor);
				}

				// Uniform across the block - indexed by j, not by thread - so
				// every lane agrees on which Gaussian it is reducing into. That
				// is what makes a warp-level sum valid here.
				const int global_id = collected_id[j];

				float2 d = { 0.f, 0.f };
				float4 con_o = { 0.f, 0.f, 0.f, 0.f };
				float G = 0.f, alpha = 0.f;
				if (active) {
					const float2 xy = collected_xy[j];
					d.x = xy.x - pixf.x;
					d.y = xy.y - pixf.y;
					con_o = collected_conic_opacity[j];
					const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
					if (power > 0.0f) {
						active = false;
					} else {
						G = DGR_EXP(power);
						alpha = min(0.99f, con_o.w * G);
						if (alpha < 1.0f / 255.0f) active = false;
					}
				}

				// This thread's contribution to each per-Gaussian gradient.
				// Zero when inactive, so an inactive lane can still take part in
				// the reduction without changing the sum.
				float g_col[C] = { 0.f };
				float g_dep = 0.f, g_m2x = 0.f, g_m2y = 0.f;
				float g_cx = 0.f, g_cy = 0.f, g_cw = 0.f, g_op = 0.f;

				if (active) {
					T /= (1.f - alpha);
					const float dchannel_dcolor = alpha * T;
					float dL_dalpha = 0.f;

					#pragma unroll
					for (int ch = 0; ch < C; ch++) {
						const float c = collected_colors[ch * BLOCK_SIZE + j];
						accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
						last_color[ch] = c;
						const float dL_dchannel = dL_dpixel[ch];
						dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
						g_col[ch] = dchannel_dcolor * dL_dchannel;
					}

					if (need_depth_grad) {
						const float depth = collected_depths[j];
						accum_rec_depth = last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
						last_depth = depth;
						dL_dalpha += (depth - accum_rec_depth) * dL_dpixel_depth;
						g_dep = dchannel_dcolor * dL_dpixel_depth;
					}

					// See the block-reduce path above for the derivation. No `skip`
					// ternary here because this whole region is already under
					// `if (active)`, which is the same predicate inverted.
					if constexpr (NEED_ALPHA_GRAD) {
						accum_rea = last_alpha + (1.f - last_alpha) * accum_rea;
						dL_dalpha += (1.f - accum_rea) * dL_dpixel_alpha;
					}

					dL_dalpha *= T;
					last_alpha = alpha;

					float bg_dot_dpixel = 0.f;
					#pragma unroll
					for (int ii = 0; ii < C; ii++)
						bg_dot_dpixel += bg_color[ii] * dL_dpixel[ii];
					dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

					const float dL_dG = con_o.w * dL_dalpha;
					const float gdx = G * d.x;
					const float gdy = G * d.y;
					const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
					const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

					g_m2x = dL_dG * dG_ddelx * ddelx_dx;
					g_m2y = dL_dG * dG_ddely * ddely_dy;
					g_cx  = -0.5f * gdx * d.x * dL_dG;
					g_cy  = -0.5f * gdx * d.y * dL_dG;
					g_cw  = -0.5f * gdy * d.y * dL_dG;
					g_op  = tracking_only ? 0.f : (G * dL_dalpha);
				}

#ifdef DGR_WARP_REDUCE
				// Combine within the warp in REGISTERS, then one lane writes.
				//
				// All 256 threads of the block target the same 10 addresses for
				// this Gaussian, so the hardware serialises them at L2 - measured
				// at 12-16 sectors per request against 4 for a coalesced access,
				// which works out to ~30% of the A100's L2 bandwidth spent on
				// reductions alone. Reducing across 32 lanes first turns 256
				// writes into 8.
				//
				// No barrier is needed: a warp executes in lockstep, so its lanes
				// are already synchronised. That is the whole advantage over the
				// block-wide reduction the pose-gradient path uses, which buys the
				// same traffic saving for ~10 block-wide barriers per Gaussian
				// (2561 per 256-Gaussian batch) and 67.6% of its stall time.
				// BALLOT FIRST. The shuffles are the expensive part and they are
				// UNCONDITIONAL: __shfl_down_sync needs all 32 lanes, so every
				// thread runs 10 gradients x 5 steps = 50 extra instructions per
				// Gaussian whether or not it contributed. The atomics they
				// replace were paid only by ACTIVE threads, and in a tile most
				// threads are inactive for most Gaussians.
				//
				// That is why the first version measured SLOWER despite the
				// contention being gone: sectors/request 12.08 -> 1.50, L2
				// traffic 8.3x down, and the kernel still went 2.622 -> 4.135 ms.
				// RED *requests* barely moved (9.21M -> 8.91M) - the instruction
				// count was never the problem, only the serialisation.
				//
				// __ballot_sync is one warp-wide instruction that all lanes
				// execute anyway, and it lets a warp with no active lane skip all
				// 50 shuffles. Lanes in a warp cover adjacent pixels, so for a
				// given Gaussian a warp tends to be entirely covered or entirely
				// not - which is exactly when this pays.
				//
				// It cannot help PARTIALLY active warps, so if the loop is
				// dominated by those, this will not recover the regression and
				// warp aggregation is simply wrong for this kernel.
				if (__ballot_sync(0xffffffffu, active) != 0u)
				{
					const bool lane0 = ((block.thread_rank() & 31u) == 0u);
					// Same deadness rule as the plain path below. Skipping the
					// shuffles too, not just the atomic: warpReduceSum is 5
					// unconditional __shfl_down_sync per channel, which the
					// ladder measured as the expensive half of this path.
					if (!skip_color_grad) {
						#pragma unroll
						for (int ch = 0; ch < C; ch++) {
							const float r = warpReduceSum(g_col[ch]);
							// Skip the write when the whole warp contributed nothing.
							// Numerically a no-op (adding zero); saves the transaction.
							if (lane0 && r != 0.f)
								atomicAdd(&dL_dcolors[global_id * C + ch], r);
						}
					}
					if (need_depth_grad) {
						const float r = warpReduceSum(g_dep);
						if (lane0 && r != 0.f) atomicAdd(&dL_ddepths[global_id], r);
					}
					const float r_m2x = warpReduceSum(g_m2x);
					const float r_m2y = warpReduceSum(g_m2y);
					const float r_cx  = warpReduceSum(g_cx);
					const float r_cy  = warpReduceSum(g_cy);
					const float r_cw  = warpReduceSum(g_cw);
					const float r_op  = warpReduceSum(g_op);
					if (lane0) {
						if (r_m2x != 0.f) atomicAdd(&dL_dmean2D[global_id].x, r_m2x);
						if (r_m2y != 0.f) atomicAdd(&dL_dmean2D[global_id].y, r_m2y);
						if (r_cx  != 0.f) atomicAdd(&dL_dconic2D[global_id].x, r_cx);
						if (r_cy  != 0.f) atomicAdd(&dL_dconic2D[global_id].y, r_cy);
						if (r_cw  != 0.f) atomicAdd(&dL_dconic2D[global_id].w, r_cw);
						if (!tracking_only && r_op != 0.f)
							atomicAdd(&dL_dopacity[global_id], r_op);
					}
				}
#else
				// Original: every active thread writes its own contribution.
				if (active) {
#ifdef DGR_STUB_GRADS
					// ---------------------------------------------------------
					// CEILING PROBE (DGR_STUB_GRADS). TIMING ONLY - THE
					// GRADIENTS THIS BUILD PRODUCES ARE GARBAGE.
					//
					// WHAT QUESTION IT ANSWERS. Tracking needs a 6-vector;
					// mapping needs ~7 floats per Gaussian. SplaTAM and
					// Gaussian-SLAM currently do tracking with the MAPPING
					// structure - scatter per-Gaussian, then let autograd chain
					// it into a pose - which is why they measure 6.74 and 15.85
					// RED sectors/request against MonoGS's 1.00, where the pose
					// gradient is block-reduced to six numbers in-kernel.
					//
					// Restructuring SplaTAM/GSLAM onto dL_dtau is a large port.
					// This bounds it first: replacing every scatter write with a
					// thread-local add measures the kernel with the accumulation
					// STRUCTURE removed but all the gradient MATH intact. The
					// duration is the FLOOR no reduction rewrite can beat. If
					// the kernel barely moves, the port cannot pay and one
					// profile has saved a week.
					//
					// WHY A SINK AND NOT A DELETION. Deleting the writes lets
					// nvcc dead-code-eliminate everything feeding them - the
					// alpha recompute, the conic math, the whole inner loop -
					// and the "ceiling" would then be the cost of an empty loop.
					// Accumulating into a register keeps every value live while
					// removing all scatter, which is exactly the variable under
					// test. The single guarded store after the loop is what
					// keeps the sink itself alive.
					stub_sink += g_m2x + g_m2y + g_cx + g_cy + g_cw + g_op;
					#pragma unroll
					for (int ch = 0; ch < C; ch++)
						stub_sink += g_col[ch];
					if (need_depth_grad)
						stub_sink += g_dep;
#else
					// DEAD DURING TRACKING WITH FROZEN, POSE-INDEPENDENT COLOURS.
					// 3 of the ~9 atomics in this path, and the accumulation as
					// a whole measured 52.4% of this kernel (DGR_STUB_GRADS).
					// The caller proves deadness per render - see
					// utils/grad_needs.colour_grad_is_dead - because it is NOT
					// universally true: SplaTAM's depth/silhouette channels are
					// pose-derived and must keep it.
					if (!skip_color_grad) {
						#pragma unroll
						for (int ch = 0; ch < C; ch++)
							atomicAdd(&dL_dcolors[global_id * C + ch], g_col[ch]);
					}
					if (need_depth_grad)
						atomicAdd(&dL_ddepths[global_id], g_dep);
					atomicAdd(&dL_dmean2D[global_id].x, g_m2x);
					atomicAdd(&dL_dmean2D[global_id].y, g_m2y);
					atomicAdd(&dL_dconic2D[global_id].x, g_cx);
					atomicAdd(&dL_dconic2D[global_id].y, g_cy);
					atomicAdd(&dL_dconic2D[global_id].w, g_cw);
					if (!tracking_only)
						atomicAdd(&dL_dopacity[global_id], g_op);
#endif
				}
#endif
			}
		}
	}

#ifdef DGR_STUB_GRADS
	// KEEPS THE SINK - AND THEREFORE THE WHOLE INNER LOOP - ALIVE.
	//
	// The guard must be one nvcc CANNOT fold away, or it deletes the store,
	// then the sink, then every gradient computation feeding it, and the probe
	// reports the cost of an empty loop as its ceiling.
	//
	// pix_id is pix.y * W + pix.x with W and H as RUNTIME kernel arguments, so
	// the compiler cannot bound it and must emit the store. At runtime it never
	// fires: 0xFFFFFFFF is 4.29e9 pixels, far beyond any image this renders.
	//
	// Deliberately NOT a comparison against a NaN or a literal float - nvcc can
	// prove those false and eliminate them. The unknowable-index form is the
	// one that survives -O3.
	if (pix_id == 0xFFFFFFFFu)
		dL_dmean2D[0].x = stub_sink;
#endif
}

void BACKWARD::preprocess(
	cudaStream_t stream,
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float* projmatrix_raw,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dtau,
	bool tracking_only,
	bool need_depth_grad)
{
	// Propagate gradients for the path of 2D conic matrix computation.
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256, 0, stream >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		dL_dtau);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256, 0, stream >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		projmatrix_raw,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dtau,
		tracking_only,
		need_depth_grad);
}

// Dispatches the four (COMPUTE_POSE_GRAD, NEED_ALPHA_GRAD) instantiations.
//
// A TEMPLATE HELPER RATHER THAN A MACRO, because every source file in this
// directory is CRLF and none of them contains a backslash-continued macro - so
// that construct is untested in this build and not worth discovering on the VM.
template <bool COMPUTE_POSE_GRAD, bool NEED_ALPHA_GRAD>
static void launch_render_backward(
	cudaStream_t stream,
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dpixels_depth,
	const float* dL_dpixels_opacity,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths,
	bool tracking_only,
	bool skip_color_grad,
	bool need_depth_grad,
	const bool* tile_mask)
{
	renderCUDABackward<NUM_CHANNELS, COMPUTE_POSE_GRAD, NEED_ALPHA_GRAD>
		<< <grid, block, 0, stream >> >(
		ranges, point_list, W, H, bg_color, means2D, conic_opacity,
		colors, depths, final_Ts, n_contrib,
		dL_dpixels, dL_dpixels_depth, dL_dpixels_opacity,
		dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors, dL_ddepths,
		tracking_only, skip_color_grad, need_depth_grad, tile_mask);
}

void BACKWARD::render(
	cudaStream_t stream,
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dpixels_depth,
	const float* dL_dpixels_opacity,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths,
	bool compute_pose_grad,
	bool tracking_only,
	bool skip_color_grad,
	bool need_depth_grad,
	bool need_alpha_grad,
	const bool* tile_mask)
{
	// Four instantiations over two compile-time flags. The (pose, alpha) pair
	// is (false, false) for SplaTAM, (true, false) for MonoGS and (false, true)
	// for Gaussian-SLAM; the fourth never launches today but costs only compile
	// time and keeps the dispatch total.
	if (compute_pose_grad) {
		if (need_alpha_grad)
			launch_render_backward<true, true>(stream, grid, block, ranges, point_list, W, H, bg_color, means2D,
			conic_opacity, colors, depths, final_Ts, n_contrib,
			dL_dpixels, dL_dpixels_depth, dL_dpixels_opacity,
			dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors, dL_ddepths,
			tracking_only, skip_color_grad, need_depth_grad, tile_mask);
		else
			launch_render_backward<true, false>(stream, grid, block, ranges, point_list, W, H, bg_color, means2D,
			conic_opacity, colors, depths, final_Ts, n_contrib,
			dL_dpixels, dL_dpixels_depth, dL_dpixels_opacity,
			dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors, dL_ddepths,
			tracking_only, skip_color_grad, need_depth_grad, tile_mask);
	} else {
		if (need_alpha_grad)
			launch_render_backward<false, true>(stream, grid, block, ranges, point_list, W, H, bg_color, means2D,
			conic_opacity, colors, depths, final_Ts, n_contrib,
			dL_dpixels, dL_dpixels_depth, dL_dpixels_opacity,
			dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors, dL_ddepths,
			tracking_only, skip_color_grad, need_depth_grad, tile_mask);
		else
			launch_render_backward<false, false>(stream, grid, block, ranges, point_list, W, H, bg_color, means2D,
			conic_opacity, colors, depths, final_Ts, n_contrib,
			dL_dpixels, dL_dpixels_depth, dL_dpixels_opacity,
			dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors, dL_ddepths,
			tracking_only, skip_color_grad, need_depth_grad, tile_mask);
	}
}