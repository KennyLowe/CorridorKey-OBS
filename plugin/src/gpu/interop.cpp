#include "gpu/interop.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>

#include <cuda_d3d11_interop.h>

namespace ck::gpu {

int cuda_device_for_d3d11(ID3D11Device *device, std::string &err)
{
	if (!device) {
		err = "no D3D11 device";
		return -1;
	}
	IDXGIDevice *dxgi = nullptr;
	if (FAILED(device->QueryInterface(__uuidof(IDXGIDevice), (void **)&dxgi))) {
		err = "D3D11 device has no IDXGIDevice";
		return -1;
	}
	IDXGIAdapter *adapter = nullptr;
	HRESULT hr = dxgi->GetAdapter(&adapter);
	dxgi->Release();
	if (FAILED(hr)) {
		err = "cannot get the D3D11 adapter";
		return -1;
	}
	DXGI_ADAPTER_DESC desc{};
	adapter->GetDesc(&desc);
	int dev = -1;
	cudaError_t ce = cudaD3D11GetDevice(&dev, adapter);
	adapter->Release();
	if (ce != cudaSuccess) {
		cudaGetLastError();
		char name[128] = {};
		WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
		err = std::string("OBS renders on \"") + name + "\", which is not a CUDA (NVIDIA) GPU: " +
		      cudaGetErrorString(ce);
		return -1;
	}
	return dev;
}

InteropTexture::~InteropTexture()
{
	if (res_)
		cudaGraphicsUnregisterResource(res_);
}

std::unique_ptr<InteropTexture> InteropTexture::register_texture(ID3D11Texture2D *tex, Use use, std::string &err)
{
	if (!tex) {
		err = "null texture";
		return nullptr;
	}
	D3D11_TEXTURE2D_DESC d{};
	tex->GetDesc(&d);
	// CUDA can't register TYPELESS resources (OBS's GS_RGBA): create interop textures as GS_RGBA_UNORM.
	if (d.Format != DXGI_FORMAT_R8G8B8A8_UNORM) {
		err = "interop texture must be R8G8B8A8_UNORM (GS_RGBA_UNORM), got DXGI format " +
		      std::to_string((int)d.Format);
		return nullptr;
	}
	std::unique_ptr<InteropTexture> t(new InteropTexture());
	t->w_ = (int)d.Width;
	t->h_ = (int)d.Height;
	t->use_ = use;
	cudaError_t ce = cudaGraphicsD3D11RegisterResource(&t->res_, tex, cudaGraphicsRegisterFlagsNone);
	if (ce != cudaSuccess) {
		cudaGetLastError();
		t->res_ = nullptr;
		err = std::string("cudaGraphicsD3D11RegisterResource: ") + cudaGetErrorString(ce);
		return nullptr;
	}
	cudaGraphicsResourceSetMapFlags(t->res_, use == Use::Read ? cudaGraphicsMapFlagsReadOnly
								    : cudaGraphicsMapFlagsWriteDiscard);
	return t;
}

cudaError_t InteropTexture::copy_to_linear(void *dst, size_t pitch, cudaStream_t s, cudaEvent_t done)
{
	cudaError_t e = cudaGraphicsMapResources(1, &res_, s);
	if (e != cudaSuccess)
		return e;
	cudaArray_t arr = nullptr;
	e = cudaGraphicsSubResourceGetMappedArray(&arr, res_, 0, 0);
	if (e == cudaSuccess)
		e = cudaMemcpy2DFromArrayAsync(dst, pitch, arr, 0, 0, (size_t)w_ * 4, (size_t)h_, cudaMemcpyDeviceToDevice, s);
	if (e == cudaSuccess && done)
		e = cudaEventRecord(done, s);
	cudaError_t u = cudaGraphicsUnmapResources(1, &res_, s);
	return e != cudaSuccess ? e : u;
}

cudaError_t InteropTexture::copy_from_linear(const void *src, size_t pitch, cudaStream_t s, cudaEvent_t done)
{
	cudaError_t e = cudaGraphicsMapResources(1, &res_, s);
	if (e != cudaSuccess)
		return e;
	cudaArray_t arr = nullptr;
	e = cudaGraphicsSubResourceGetMappedArray(&arr, res_, 0, 0);
	if (e == cudaSuccess)
		e = cudaMemcpy2DToArrayAsync(arr, 0, 0, src, pitch, (size_t)w_ * 4, (size_t)h_, cudaMemcpyDeviceToDevice, s);
	if (e == cudaSuccess && done)
		e = cudaEventRecord(done, s);
	cudaError_t u = cudaGraphicsUnmapResources(1, &res_, s);
	return e != cudaSuccess ? e : u;
}

} // namespace ck::gpu
