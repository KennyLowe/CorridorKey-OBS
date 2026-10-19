// Runtime loading of TensorRT and NVML, and the GPU identity used in the engine cache key.
//
// nvinfer_10.dll is delay-loaded: it is loaded here by absolute path from the configured TensorRT
// folder (LoadLibraryExW with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
// no SetDllDirectory, research D2), then every delay-load import is bound up front with
// __HrLoadAllImportsForDll. Any failure is reported as an error string ("unsupported"), never a
// crash: no TensorRT function is called unless this succeeded.
#pragma once

#include <string>

namespace ck::gpu {

// Thread-safe, idempotent. Returns "" on success, else why TensorRT can't be used.
// A different folder than a previous successful call is refused (one nvinfer per process).
std::string load_tensorrt(const std::string &dir_utf8);
bool tensorrt_loaded();
// "10.16.1.11" when the loaded runtime is 10.16.1 (the build number comes from the headers the
// plugin was compiled with); "" if not loaded.
std::string tensorrt_version();
// The only TensorRT the plugin accepts (engines are tied to the exact version).
constexpr const char *kRequiredTrtVersion = "10.16.1.11";

struct DeviceInfo {
	int cuda_device = -1;
	std::string name;           // "NVIDIA GeForce RTX 3060"
	std::string uuid;           // "GPU-0763dee5-2e3a-a1b2-955d-4600c81fd6c0" (NVML format)
	std::string driver_version; // "576.80" (NVML), "" if NVML is unavailable
	int sm_major = 0, sm_minor = 0;
};

// Identity of a CUDA device; NVML (nvml.dll from System32) supplies the driver string.
DeviceInfo device_info(int cuda_device);

} // namespace ck::gpu
