#include "gpu/dll_loader.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <delayimp.h>

#include <cstdio>
#include <filesystem>
#include <mutex>

#include <cuda_runtime.h>
#include <NvInferRuntime.h>

namespace ck::gpu {
namespace {

std::mutex g_mu;
HMODULE g_nvinfer = nullptr;
std::string g_dir;

std::wstring widen(const std::string &s)
{
	if (s.empty())
		return {};
	int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
	std::wstring w(n, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
	return w;
}

std::string win_error(DWORD e)
{
	char buf[256] = {};
	FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, e, 0, buf, sizeof(buf),
		       nullptr);
	std::string s(buf);
	while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
		s.pop_back();
	return s + " (" + std::to_string(e) + ")";
}

// SEH can't live in a function with C++ objects to unwind, so it gets its own.
HRESULT bind_nvinfer_imports()
{
	__try {
		return __HrLoadAllImportsForDll("nvinfer_10.dll");
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		return E_FAIL;
	}
}

} // namespace

std::string load_tensorrt(const std::string &dir_utf8)
{
	std::lock_guard lk(g_mu);
	namespace fs = std::filesystem;
	fs::path dir = fs::path(widen(dir_utf8));
	fs::path dll = dir / L"nvinfer_10.dll";
	std::error_code ec;
	if (!fs::exists(dll, ec))
		return "nvinfer_10.dll not found in " + dir_utf8;
	if (g_nvinfer) {
		if (fs::equivalent(dir, fs::path(widen(g_dir)), ec))
			return {};
		return "TensorRT is already loaded from " + g_dir + "; restart OBS to use " + dir_utf8;
	}
	HMODULE h = LoadLibraryExW(dll.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
	if (!h)
		return "failed to load " + dll.string() + ": " + win_error(GetLastError());
	// Bind every delay-load import now so a missing export fails here, not mid-stream.
	if (FAILED(bind_nvinfer_imports())) {
		FreeLibrary(h);
		return "nvinfer_10.dll in " + dir_utf8 + " is missing TensorRT runtime exports";
	}
	int32_t v = getInferLibVersion();
	if (v != NV_TENSORRT_VERSION) {
		// Keep it loaded (bound imports point into it) but refuse to use it.
		g_nvinfer = h;
		g_dir = dir_utf8;
		char buf[96];
		snprintf(buf, sizeof(buf), "TensorRT %d.%d.%d found, %s required", v / 10000, (v / 100) % 100, v % 100,
			 kRequiredTrtVersion);
		return buf;
	}
	g_nvinfer = h;
	g_dir = dir_utf8;
	return {};
}

bool tensorrt_loaded()
{
	std::lock_guard lk(g_mu);
	return g_nvinfer != nullptr && getInferLibVersion() == NV_TENSORRT_VERSION;
}

std::string tensorrt_version()
{
	if (!tensorrt_loaded())
		return {};
	int32_t v = getInferLibVersion();
	char buf[64];
	snprintf(buf, sizeof(buf), "%d.%d.%d.%d", v / 10000, (v / 100) % 100, v % 100, NV_TENSORRT_BUILD);
	return buf;
}

// ------------------------------------------------------------------------------- NVML (driver)

namespace {
using nvmlReturn_t = int;
using nvmlDevice_t = void *;
struct Nvml {
	bool ok = false;
	nvmlReturn_t(__cdecl *init)() = nullptr;
	nvmlReturn_t(__cdecl *driver)(char *, unsigned) = nullptr;
};
Nvml &nvml()
{
	static Nvml n = [] {
		Nvml r;
		HMODULE h = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
		if (!h)
			return r;
		r.init = (nvmlReturn_t(__cdecl *)())GetProcAddress(h, "nvmlInit_v2");
		r.driver = (nvmlReturn_t(__cdecl *)(char *, unsigned))GetProcAddress(h, "nvmlSystemGetDriverVersion");
		r.ok = r.init && r.driver && r.init() == 0;
		return r;
	}();
	return n;
}
} // namespace

DeviceInfo device_info(int cuda_device)
{
	DeviceInfo d;
	cudaDeviceProp p{};
	if (cudaGetDeviceProperties(&p, cuda_device) != cudaSuccess)
		return d;
	d.cuda_device = cuda_device;
	d.name = p.name;
	d.sm_major = p.major;
	d.sm_minor = p.minor;
	const unsigned char *u = (const unsigned char *)p.uuid.bytes;
	char buf[64];
	snprintf(buf, sizeof(buf), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1],
		 u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
	d.uuid = buf;
	if (nvml().ok) {
		char v[96] = {};
		if (nvml().driver(v, sizeof(v)) == 0)
			d.driver_version = v;
	}
	return d;
}

} // namespace ck::gpu
