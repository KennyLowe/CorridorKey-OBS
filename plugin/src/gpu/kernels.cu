#include "gpu/kernels.cuh"

#include <algorithm>
#include <cmath>

namespace ck::gpu {
namespace {

__constant__ float kMean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float kStd[3] = {0.229f, 0.224f, 0.225f};
constexpr int kBlur = 5;
// Max taps of the antialiased filter: support = scale (downscale), 2*support+1 inputs.
constexpr int kMaxTaps = 64;

__device__ __forceinline__ float smoothstep_mask(float d, float lo, float hi)
{
	float t = fminf(fmaxf((d - lo) / (hi - lo), 0.0f), 1.0f);
	return 1.0f - t * t * (3.0f - 2.0f * t);
}

__global__ void k_hint_raw(const uint8_t *rgba, size_t pitch, int W, int H, HintParams p, float *out)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H)
		return;
	const uint8_t *px = rgba + (size_t)y * pitch + (size_t)x * 4;
	float r = px[0] / 255.0f, g = px[1] / 255.0f, b = px[2] / 255.0f;
	float d = (g - fmaxf(r, b)) / p.kd;
	out[(size_t)y * W + x] = smoothstep_mask(d, p.lo, p.hi);
}

__global__ void k_box_blur(const float *in, int W, int H, float *out)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H)
		return;
	const int r = kBlur / 2;
	float sum = 0.0f;
	int n = 0;
	for (int dy = -r; dy <= r; dy++) {
		int yy = y + dy;
		if (yy < 0 || yy >= H)
			continue;
		for (int dx = -r; dx <= r; dx++) {
			int xx = x + dx;
			if (xx < 0 || xx >= W)
				continue;
			sum += in[(size_t)yy * W + xx];
			n++;
		}
	}
	out[(size_t)y * W + x] = fminf(fmaxf(sum / (float)n, 0.0f), 1.0f);
}

// PyTorch's antialiased bilinear weights (UpSampleKernel.cpp, _compute_indices_min_size_weights_aa)
// for output index i: returns xmin and the number of taps, writes normalised weights.
__device__ __forceinline__ int aa_weights(int i, int in_size, int out_size, int &xmin, float *w)
{
	float scale = (float)in_size / (float)out_size;
	float support = scale >= 1.0f ? scale : 1.0f;
	float invscale = scale >= 1.0f ? 1.0f / scale : 1.0f;
	float center = scale * ((float)i + 0.5f);
	xmin = max((int)(center - support + 0.5f), 0);
	int xsize = min((int)(center + support + 0.5f), in_size) - xmin;
	xsize = min(xsize, kMaxTaps);
	float total = 0.0f;
	for (int j = 0; j < xsize; j++) {
		float v = fmaxf(0.0f, 1.0f - fabsf(((float)(j + xmin) - center + 0.5f) * invscale));
		w[j] = v;
		total += v;
	}
	if (total != 0.0f)
		for (int j = 0; j < xsize; j++)
			w[j] /= total;
	return xsize;
}

// Horizontal pass: rgba8 (RGB) + mask, W x H -> 4 planes of w x H floats in [0,1].
__global__ void k_aa_horizontal(const uint8_t *rgba, size_t pitch, const float *mask, int W, int H, int w,
				float *tmp)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= H)
		return;
	float wt[kMaxTaps];
	int xmin;
	int n = aa_weights(x, W, w, xmin, wt);
	const uint8_t *row = rgba + (size_t)y * pitch;
	const float *mrow = mask + (size_t)y * W;
	float acc[4] = {0, 0, 0, 0};
	for (int j = 0; j < n; j++) {
		const uint8_t *px = row + (size_t)(xmin + j) * 4;
		acc[0] += wt[j] * (px[0] / 255.0f);
		acc[1] += wt[j] * (px[1] / 255.0f);
		acc[2] += wt[j] * (px[2] / 255.0f);
		acc[3] += wt[j] * mrow[xmin + j];
	}
	size_t plane = (size_t)w * H;
	for (int c = 0; c < 4; c++)
		tmp[c * plane + (size_t)y * w + x] = acc[c];
}

// Vertical pass: 4 planes w x H -> NCHW [1,4,h,w], RGB normalised with ImageNet mean/std.
__global__ void k_aa_vertical(const float *tmp, int H, int w, int h, float *out)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h)
		return;
	float wt[kMaxTaps];
	int ymin;
	int n = aa_weights(y, H, h, ymin, wt);
	size_t in_plane = (size_t)w * H, out_plane = (size_t)w * h;
	for (int c = 0; c < 4; c++) {
		float acc = 0.0f;
		for (int j = 0; j < n; j++)
			acc += wt[j] * tmp[c * in_plane + (size_t)(ymin + j) * w + x];
		if (c < 3)
			acc = (acc - kMean[c]) / kStd[c];
		out[c * out_plane + (size_t)y * w + x] = acc;
	}
}

// Bilinear, align_corners=False, no antialias (F.interpolate): source coordinate and taps.
__device__ __forceinline__ void lin_coord(int dst, int in_size, int out_size, int &i0, int &i1, float &l1)
{
	float scale = (float)in_size / (float)out_size;
	float src = ((float)dst + 0.5f) * scale - 0.5f;
	if (src < 0.0f)
		src = 0.0f;
	i0 = (int)src;
	i1 = i0 + (i0 < in_size - 1 ? 1 : 0);
	l1 = src - (float)i0;
}

__device__ __forceinline__ uint8_t to_u8(float v)
{
	return (uint8_t)fminf(fmaxf(rintf(v * 255.0f), 0.0f), 255.0f);
}

__global__ void k_postprocess(const float *alpha, const float *fg, int w, int h, uint8_t *out, size_t pitch, int W,
			      int H)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H)
		return;
	int x0, x1, y0, y1;
	float lx, ly;
	lin_coord(x, w, W, x0, x1, lx);
	lin_coord(y, h, H, y0, y1, ly);
	size_t plane = (size_t)w * h;
	auto sample = [&](const float *p) {
		float a = p[(size_t)y0 * w + x0], b = p[(size_t)y0 * w + x1];
		float c = p[(size_t)y1 * w + x0], d = p[(size_t)y1 * w + x1];
		return (1.0f - ly) * ((1.0f - lx) * a + lx * b) + ly * ((1.0f - lx) * c + lx * d);
	};
	uint8_t *px = out + (size_t)y * pitch + (size_t)x * 4;
	px[0] = to_u8(sample(fg));
	px[1] = to_u8(sample(fg + plane));
	px[2] = to_u8(sample(fg + 2 * plane));
	px[3] = to_u8(sample(alpha));
}

__global__ void k_mask_to_rgba(const float *mask, int W, int H, uint8_t *out, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H)
		return;
	uint8_t v = to_u8(mask[(size_t)y * W + x]);
	uint8_t *px = out + (size_t)y * pitch + (size_t)x * 4;
	px[0] = px[1] = px[2] = v;
	px[3] = 255;
}

dim3 grid_for(int w, int h, dim3 b)
{
	return dim3((unsigned)((w + b.x - 1) / b.x), (unsigned)((h + b.y - 1) / b.y));
}

} // namespace

size_t preprocess_scratch_floats(int /*W*/, int H, int w, int /*h*/)
{
	return (size_t)4 * w * H;
}

cudaError_t hint(const uint8_t *rgba, size_t pitch, int W, int H, HintParams p, float *tmp, float *mask,
		 cudaStream_t s)
{
	if (!(p.hi > p.lo) || p.kd <= 0.0f)
		return cudaErrorInvalidValue;
	dim3 b(32, 8);
	k_hint_raw<<<grid_for(W, H, b), b, 0, s>>>(rgba, pitch, W, H, p, tmp);
	k_box_blur<<<grid_for(W, H, b), b, 0, s>>>(tmp, W, H, mask);
	return cudaGetLastError();
}

cudaError_t preprocess(const uint8_t *rgba, size_t pitch, const float *mask, int W, int H, float *out, int w, int h,
		       float *scratch, cudaStream_t s)
{
	// Downscale only (internal sizes are below camera sizes). Upscaling is supported by the same
	// weights (support clamps to 1) but the tap limit bounds the downscale factor.
	if ((float)W / (float)w * 2.0f + 2.0f > (float)kMaxTaps || (float)H / (float)h * 2.0f + 2.0f > (float)kMaxTaps)
		return cudaErrorInvalidValue;
	dim3 b(32, 8);
	k_aa_horizontal<<<grid_for(w, H, b), b, 0, s>>>(rgba, pitch, mask, W, H, w, scratch);
	k_aa_vertical<<<grid_for(w, h, b), b, 0, s>>>(scratch, H, w, h, out);
	return cudaGetLastError();
}

cudaError_t postprocess(const float *alpha, const float *fg, int w, int h, uint8_t *out, size_t pitch, int W, int H,
			cudaStream_t s)
{
	dim3 b(32, 8);
	k_postprocess<<<grid_for(W, H, b), b, 0, s>>>(alpha, fg, w, h, out, pitch, W, H);
	return cudaGetLastError();
}

cudaError_t mask_to_rgba(const float *mask, int W, int H, uint8_t *out, size_t pitch, cudaStream_t s)
{
	dim3 b(32, 8);
	k_mask_to_rgba<<<grid_for(W, H, b), b, 0, s>>>(mask, W, H, out, pitch);
	return cudaGetLastError();
}

} // namespace ck::gpu
