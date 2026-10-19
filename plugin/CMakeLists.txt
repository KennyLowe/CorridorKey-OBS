cmake_minimum_required(VERSION 3.28...3.30)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version})

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)
option(ENABLE_QT "Use Qt functionality" OFF)
option(CK_BUILD_TESTS "Build the Catch2 unit and GPU tests" ON)

include(compilerconfig)
include(defaults)
include(helpers)

# corridorkey-obs: C++20, CUDA 12.9 (sm_86: RTX 3060 and RTX A2000 are both Ampere GA10x),
# static cudart, TensorRT runtime only (research D2).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_STANDARD 20)
enable_language(CUDA)
find_package(CUDAToolkit 12.9 REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT 10.16 REQUIRED)

# ---------------------------------------------------------------- third party (licences checked)
# nlohmann/json 3.12.0, MIT licence (GPL-compatible), header-only.
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
)
FetchContent_MakeAvailable(nlohmann_json)

# ---------------------------------------------------------------- ck_core: pure C++, no OBS/CUDA
add_library(
  ck_core
  STATIC
  src/core/state.cpp
  src/core/scheduler.cpp
  src/core/stats.cpp
  src/core/cache_key.cpp
  src/core/sha256.cpp
  src/core/auto_select.cpp
)
target_include_directories(ck_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(ck_core PUBLIC nlohmann_json::nlohmann_json)

# ---------------------------------------------------------------- ck_gpu: CUDA kernels, TensorRT, D3D11 interop
add_library(
  ck_gpu
  STATIC
  src/gpu/kernels.cu
  src/gpu/trt_runner.cpp
  src/gpu/interop.cpp
  src/gpu/dll_loader.cpp
)
target_include_directories(ck_gpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(ck_gpu PUBLIC CUDA::cudart_static TensorRT::nvinfer d3d11 dxgi ck_core)
# cudart_static.lib names LIBCMT as a default library; everything else here uses the /MD CRT.
target_link_options(ck_gpu INTERFACE /NODEFAULTLIB:LIBCMT)

# ---------------------------------------------------------------- the OBS module
add_library(${CMAKE_PROJECT_NAME} MODULE)

find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs ck_core ck_gpu delayimp)

if(ENABLE_FRONTEND_API)
  find_package(obs-frontend-api REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

if(ENABLE_QT)
  find_package(Qt6 COMPONENTS Widgets Core)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets)
  target_compile_options(
    ${CMAKE_PROJECT_NAME}
    PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-quoted-include-in-framework-header -Wno-comma>
  )
  set_target_properties(
    ${CMAKE_PROJECT_NAME}
    PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON
  )
endif()

target_sources(
  ${CMAKE_PROJECT_NAME}
  PRIVATE src/plugin-main.cpp src/ck-filter.cpp src/worker.cpp src/prepare_client.cpp src/ck-log.cpp
)

# Default helper command for this checkout (owner-only v1): uv run --project <repo>/bench ckbench
cmake_path(SET _ck_bench_dir NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../bench")
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE CK_BENCH_DIR="${_ck_bench_dir}")

# TensorRT is loaded at runtime from the configured folder (LoadLibraryExW with an absolute path);
# the import is delay-loaded so OBS never fails to load the module when TensorRT is missing.
target_link_options(${CMAKE_PROJECT_NAME} PRIVATE /DELAYLOAD:nvinfer_10.dll /IGNORE:4199)

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})

# ---------------------------------------------------------------- tests
if(CK_BUILD_TESTS)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(Catch2)
  # The template defines _UNICODE globally, which makes Catch2's main a wmain; keep a plain main.
  target_compile_definitions(Catch2WithMain PRIVATE DO_NOT_USE_WMAIN)
  enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()
