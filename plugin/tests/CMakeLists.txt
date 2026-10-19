# Catch2 v3 test executables. ctest runs both; the GPU suite needs an NVIDIA GPU, CUDA and
# TensorRT (CK_TRT_DIR, default C:/TensorRT-10.16.1.11/bin).
include(Catch)

add_executable(ck_unit_tests unit/test_smoke.cpp)
target_link_libraries(ck_unit_tests PRIVATE ck_core Catch2::Catch2WithMain)
target_compile_definitions(ck_unit_tests PRIVATE CK_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
catch_discover_tests(ck_unit_tests TEST_PREFIX "unit:" DISCOVERY_MODE PRE_TEST)

add_executable(ck_gpu_tests gpu/test_gpu_smoke.cpp)
target_link_libraries(ck_gpu_tests PRIVATE ck_gpu ck_core Catch2::Catch2WithMain delayimp)
target_link_options(ck_gpu_tests PRIVATE /DELAYLOAD:nvinfer_10.dll /IGNORE:4199)
set(CK_TRT_DIR "C:/TensorRT-10.16.1.11/bin" CACHE PATH "TensorRT DLL folder used by the GPU tests")
target_compile_definitions(ck_gpu_tests PRIVATE CK_TRT_DIR="${CK_TRT_DIR}" CK_REPO_ROOT="${CMAKE_SOURCE_DIR}/..")
catch_discover_tests(ck_gpu_tests TEST_PREFIX "gpu:" DISCOVERY_MODE PRE_TEST)
