// D3D11 <-> CUDA texture interop (research D4). Everything here runs on OBS's graphics thread
// (map/unmap touch the D3D11 immediate context, which is not thread-safe); the worker only ever
// sees linear device buffers.
//
// Steady state is asynchronous: map -> cudaMemcpy2D{From,To}ArrayAsync on a non-blocking stream ->
// optional event record -> unmap. Registration happens once per texture (re-registered on resize).
#pragma once

#include <memory>
#include <string>

#include <cuda_runtime.h>

struct ID3D11Device;
struct ID3D11Texture2D;
struct cudaGraphicsResource;

namespace ck::gpu {

// The CUDA device that drives the D3D11 device's adapter, or -1 with `err` (e.g. OBS is running
// on a non-NVIDIA adapter, or no CUDA device exists).
int cuda_device_for_d3d11(ID3D11Device *device, std::string &err);

class InteropTexture {
public:
	enum class Use { Read, Write };
	~InteropTexture();
	// The texture must be R8G8B8A8_UNORM (OBS: GS_RGBA_UNORM) with one mip level. CUDA can't
	// register OBS's TYPELESS GS_RGBA textures; gs_copy_texture between the two is allowed.
	static std::unique_ptr<InteropTexture> register_texture(ID3D11Texture2D *tex, Use use, std::string &err);

	int width() const { return w_; }
	int height() const { return h_; }

	// Texture -> linear RGBA8 buffer (pitch in bytes). Records `done` (if given) after the copy.
	cudaError_t copy_to_linear(void *dst, size_t pitch, cudaStream_t s, cudaEvent_t done = nullptr);
	// Linear RGBA8 buffer -> texture. Records `done` (if given) after the copy.
	cudaError_t copy_from_linear(const void *src, size_t pitch, cudaStream_t s, cudaEvent_t done = nullptr);

private:
	InteropTexture() = default;
	cudaGraphicsResource *res_ = nullptr;
	int w_ = 0, h_ = 0;
	Use use_ = Use::Read;
};

} // namespace ck::gpu
