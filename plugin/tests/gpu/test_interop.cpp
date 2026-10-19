// D3D11 <-> CUDA interop round trip on our own D3D11 device (T016).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>

#include <cuda_d3d11_interop.h>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <vector>

#include "gpu/interop.hpp"

using namespace ck::gpu;

namespace {

struct D3D {
	ID3D11Device *dev = nullptr;
	ID3D11DeviceContext *ctx = nullptr;
	~D3D()
	{
		if (ctx)
			ctx->Release();
		if (dev)
			dev->Release();
	}
};

// A D3D11 device on the first adapter CUDA can drive (as OBS would be on the NVIDIA GPU).
bool make_device(D3D &d)
{
	IDXGIFactory1 *f = nullptr;
	if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&f)))
		return false;
	IDXGIAdapter1 *a = nullptr;
	bool ok = false;
	for (UINT i = 0; !ok && f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
		int cd = -1;
		if (cudaD3D11GetDevice(&cd, a) == cudaSuccess) {
			D3D_FEATURE_LEVEL fl;
			ok = SUCCEEDED(D3D11CreateDevice(a, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
							 &d.dev, &fl, &d.ctx));
		}
		cudaGetLastError();
		a->Release();
	}
	f->Release();
	return ok;
}

ID3D11Texture2D *make_tex(ID3D11Device *dev, int w, int h, DXGI_FORMAT fmt, const std::vector<uint8_t> *init)
{
	D3D11_TEXTURE2D_DESC td{};
	td.Width = w;
	td.Height = h;
	td.MipLevels = 1;
	td.ArraySize = 1;
	td.Format = fmt;
	td.SampleDesc.Count = 1;
	td.Usage = D3D11_USAGE_DEFAULT;
	td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	D3D11_SUBRESOURCE_DATA sd{};
	if (init) {
		sd.pSysMem = init->data();
		sd.SysMemPitch = w * 4;
	}
	ID3D11Texture2D *t = nullptr;
	dev->CreateTexture2D(&td, init ? &sd : nullptr, &t);
	return t;
}

std::vector<uint8_t> read_back(D3D &d, ID3D11Texture2D *tex, int w, int h, DXGI_FORMAT fmt)
{
	D3D11_TEXTURE2D_DESC td{};
	tex->GetDesc(&td);
	td.Usage = D3D11_USAGE_STAGING;
	td.BindFlags = 0;
	td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	td.Format = fmt;
	ID3D11Texture2D *st = nullptr;
	d.dev->CreateTexture2D(&td, nullptr, &st);
	d.ctx->CopyResource(st, tex);
	D3D11_MAPPED_SUBRESOURCE m{};
	d.ctx->Map(st, 0, D3D11_MAP_READ, 0, &m);
	std::vector<uint8_t> out((size_t)w * h * 4);
	for (int y = 0; y < h; y++)
		memcpy(out.data() + (size_t)y * w * 4, (uint8_t *)m.pData + (size_t)y * m.RowPitch, (size_t)w * 4);
	d.ctx->Unmap(st, 0);
	st->Release();
	return out;
}

std::vector<uint8_t> pattern(int w, int h, int seed)
{
	std::vector<uint8_t> p((size_t)w * h * 4);
	for (size_t i = 0; i < p.size(); i++)
		p[i] = (uint8_t)((i * 31 + seed * 7 + (i / 4) % 13) & 0xff);
	return p;
}

} // namespace

TEST_CASE("D3D11 texture <-> CUDA linear buffer round trip", "[interop]")
{
	D3D d;
	REQUIRE(make_device(d));
	std::string err;
	int cd = cuda_device_for_d3d11(d.dev, err);
	INFO(err);
	REQUIRE(cd >= 0);
	REQUIRE(cudaSetDevice(cd) == cudaSuccess);

	for (DXGI_FORMAT fmt : {DXGI_FORMAT_R8G8B8A8_UNORM}) {
		const int w = 643, h = 487; // odd size: pitch handling
		auto src = pattern(w, h, 1);
		ID3D11Texture2D *in_tex = make_tex(d.dev, w, h, fmt, &src);
		ID3D11Texture2D *out_tex = make_tex(d.dev, w, h, fmt, nullptr);
		REQUIRE(in_tex);
		REQUIRE(out_tex);
		auto in = InteropTexture::register_texture(in_tex, InteropTexture::Use::Read, err);
		INFO(err);
		REQUIRE(in);
		auto out = InteropTexture::register_texture(out_tex, InteropTexture::Use::Write, err);
		REQUIRE(out);
		CHECK(in->width() == w);
		CHECK(in->height() == h);

		cudaStream_t s;
		REQUIRE(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);
		cudaEvent_t ev;
		REQUIRE(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess);
		void *lin = nullptr;
		size_t pitch = 0;
		REQUIRE(cudaMallocPitch(&lin, &pitch, (size_t)w * 4, h) == cudaSuccess);

		// texture -> linear
		REQUIRE(in->copy_to_linear(lin, pitch, s, ev) == cudaSuccess);
		REQUIRE(cudaEventSynchronize(ev) == cudaSuccess);
		std::vector<uint8_t> got((size_t)w * h * 4);
		REQUIRE(cudaMemcpy2D(got.data(), (size_t)w * 4, lin, pitch, (size_t)w * 4, h, cudaMemcpyDeviceToHost) ==
			cudaSuccess);
		CHECK(got == src);

		// linear -> texture (a different pattern), read back through a staging texture
		auto src2 = pattern(w, h, 2);
		REQUIRE(cudaMemcpy2D(lin, pitch, src2.data(), (size_t)w * 4, (size_t)w * 4, h, cudaMemcpyHostToDevice) ==
			cudaSuccess);
		REQUIRE(out->copy_from_linear(lin, pitch, s, ev) == cudaSuccess);
		REQUIRE(cudaEventSynchronize(ev) == cudaSuccess);
		CHECK(read_back(d, out_tex, w, h, fmt) == src2);

		// steady-state cost of one map/copy/unmap on the calling thread (informational)
		auto t0 = std::chrono::steady_clock::now();
		for (int i = 0; i < 100; i++)
			REQUIRE(in->copy_to_linear(lin, pitch, s, ev) == cudaSuccess);
		auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 100;
		REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
		WARN("format " << (int)fmt << ": map+copy+unmap issue cost " << us << " us");

		cudaFree(lin);
		cudaEventDestroy(ev);
		cudaStreamDestroy(s);
		in.reset();
		out.reset();
		in_tex->Release();
		out_tex->Release();
	}
}

TEST_CASE("OBS's TYPELESS GS_RGBA textures are refused with a clear error", "[interop]")
{
	D3D d;
	REQUIRE(make_device(d));
	ID3D11Texture2D *t = make_tex(d.dev, 64, 64, DXGI_FORMAT_R8G8B8A8_TYPELESS, nullptr);
	REQUIRE(t);
	std::string err;
	CHECK_FALSE(InteropTexture::register_texture(t, InteropTexture::Use::Read, err));
	CHECK(err.find("GS_RGBA_UNORM") != std::string::npos);
	t->Release();
}

TEST_CASE("a null device is reported, not a crash", "[interop]")
{
	std::string err;
	CHECK(cuda_device_for_d3d11(nullptr, err) == -1);
	CHECK_FALSE(err.empty());
}
