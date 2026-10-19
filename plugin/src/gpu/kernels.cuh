// CUDA kernels for the keying pipeline (research D4, D6). Host-callable launchers; all work is
// queued on the given stream (never the default stream) and nothing here synchronises.
//
// Matches the feature 001 spike (bench/ckbench/hint.py, pipeline.py):
//   hint:        d = (g - max(r,b)) / kd; t = clamp((d - lo) / (hi - lo)); m = 1 - t^2 (3 - 2t);
//                then a 5x5 box blur that averages only in-image pixels (count_include_pad=False).
//                Fixed key: kd = 1, lo = 0.05, hi = 0.15 (chroma_hint). Calibrated/manual:
//                kd = screen g - max(screen r, b), lo/hi user (default 0.30/0.70).
//   preprocess:  RGB and hint downscaled to the internal size with antialiased bilinear
//                (torchvision resize(antialias=True)), RGB normalised with ImageNet mean/std,
//                hint not normalised -> float32 NCHW [1,4,h,w].
//   postprocess: alpha [1,1,h,w] and fg [1,3,h,w] upscaled with bilinear (align_corners=False,
//                no antialias) to the camera size, packed RGBA8 = (fg.rgb, alpha), x255, rounded
//                half-to-even, clamped.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ck::gpu {

struct HintParams {
	float kd = 1.0f;
	float lo = 0.05f;
	float hi = 0.15f;
};

// Scratch needed by preprocess for a W x H camera frame and a w x h internal size (floats).
size_t preprocess_scratch_floats(int W, int H, int w, int h);

// rgba: device RGBA8, row pitch in bytes. tmp/mask: W*H floats each.
cudaError_t hint(const uint8_t *rgba, size_t pitch, int W, int H, HintParams p, float *tmp, float *mask,
		 cudaStream_t s);

// rgba + mask (camera res) -> out [1,4,h,w] float32.
cudaError_t preprocess(const uint8_t *rgba, size_t pitch, const float *mask, int W, int H, float *out, int w, int h,
		       float *scratch, cudaStream_t s);

// alpha [h,w], fg [3,h,w] float32 -> out RGBA8 W x H with row pitch in bytes.
cudaError_t postprocess(const float *alpha, const float *fg, int w, int h, uint8_t *out, size_t pitch, int W, int H,
			cudaStream_t s);

// Fills an RGBA8 buffer with the source frame's alpha replaced by the hint mask (show-hint view
// is drawn by the colour-key shader instead; this is only used by tests and diagnostics).
cudaError_t mask_to_rgba(const float *mask, int W, int H, uint8_t *out, size_t pitch, cudaStream_t s);

} // namespace ck::gpu
