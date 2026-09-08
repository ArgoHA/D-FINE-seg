cmake_minimum_required(VERSION 3.22)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)  # RTX 50xx; override with -DCMAKE_CUDA_ARCHITECTURES=89 for Ada etc.
endif()
project(dfine_e2e LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(AV REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)

# TensorRT: headers from the system package (libnvinfer-headers-dev), runtime from the
# venv wheel so the binary runs the exact version that built the engines.
find_path(TRT_INCLUDE_DIR NvInfer.h PATHS /usr/include/x86_64-linux-gnu /usr/include REQUIRED)
file(GLOB TRT_VENV_DIRS ${CMAKE_SOURCE_DIR}/../.venv/lib/python3.*/site-packages/tensorrt_libs)
find_library(TRT_LIB NAMES libnvinfer.so.10 nvinfer HINTS ${TRT_LIB_DIR} ${TRT_VENV_DIRS} REQUIRED)
get_filename_component(TRT_LIB_DIR_RESOLVED ${TRT_LIB} DIRECTORY)

add_executable(dfine_e2e
  src/main.cpp src/trt_model.cpp src/nvdec.cpp src/nvenc.cpp src/kernels.cu)
target_include_directories(dfine_e2e PRIVATE src)
target_include_directories(dfine_e2e SYSTEM PRIVATE third_party ${TRT_INCLUDE_DIR})
target_link_libraries(dfine_e2e PRIVATE
  ${TRT_LIB} CUDA::cudart CUDA::cuda_driver PkgConfig::AV Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(dfine_e2e PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>)
set_target_properties(dfine_e2e PROPERTIES
  BUILD_RPATH "${TRT_LIB_DIR_RESOLVED};${CUDAToolkit_LIBRARY_DIR}")
