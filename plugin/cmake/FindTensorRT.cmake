# FindTensorRT: locate the TensorRT 10 runtime SDK (headers + nvinfer_10 import library).
#
# Input:  TensorRT_ROOT (cache/env), e.g. C:/TensorRT-10.16.1.11
# Output: TensorRT_FOUND, TensorRT_INCLUDE_DIR, TensorRT_LIBRARY, TensorRT_VERSION,
#         imported target TensorRT::nvinfer
#
# Only the runtime (nvinfer) is used by the plugin: no ONNX parser and no builder (research D2).

if(NOT TensorRT_ROOT AND DEFINED ENV{TensorRT_ROOT})
  set(TensorRT_ROOT "$ENV{TensorRT_ROOT}")
endif()
set(TensorRT_ROOT "${TensorRT_ROOT}" CACHE PATH "TensorRT SDK root (contains include/, lib/, bin/)")

find_path(TensorRT_INCLUDE_DIR NvInfer.h HINTS "${TensorRT_ROOT}" PATH_SUFFIXES include)
find_library(TensorRT_LIBRARY NAMES nvinfer_10 nvinfer HINTS "${TensorRT_ROOT}" PATH_SUFFIXES lib lib64)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_ver REGEX "#define TRT_(MAJOR|MINOR|PATCH|BUILD)_ENTERPRISE ")
  if(NOT _trt_ver)
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_ver REGEX "#define NV_TENSORRT_(MAJOR|MINOR|PATCH|BUILD) ")
  endif()
  foreach(_part MAJOR MINOR PATCH BUILD)
    string(REGEX MATCH "_${_part}[A-Z_]* +([0-9]+)" _m "${_trt_ver}")
    set(_trt_${_part} "${CMAKE_MATCH_1}")
  endforeach()
  set(TensorRT_VERSION "${_trt_MAJOR}.${_trt_MINOR}.${_trt_PATCH}.${_trt_BUILD}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TensorRT
  REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_LIBRARY
  VERSION_VAR TensorRT_VERSION
)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
  add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
  set_target_properties(
    TensorRT::nvinfer
    PROPERTIES IMPORTED_LOCATION "${TensorRT_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY)
