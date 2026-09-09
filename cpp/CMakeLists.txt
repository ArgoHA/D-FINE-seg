cmake_minimum_required(VERSION 3.22)
# Default targets the dev box (RTX 50xx). Jetson Orin = 87, Ada = 89, Ampere = 86; a build for
# one arch does NOT run on an older one (the embedded PTX only JITs forward), so set this to
# the deployment GPU: -DCMAKE_CUDA_ARCHITECTURES=87
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
project(dfine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(DFINE_BUILD_VIDEO "Build the NVDEC/NVENC end-to-end video app" ON)
option(DFINE_BUILD_EXAMPLES "Build the example consumers" ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# TensorRT: headers from the system package (libnvinfer-headers-dev), runtime from the
# venv wheel so the binary runs the exact version that built the engines.
find_path(TRT_INCLUDE_DIR NvInfer.h PATHS /usr/include/x86_64-linux-gnu /usr/include REQUIRED)
file(GLOB TRT_VENV_DIRS ${CMAKE_SOURCE_DIR}/../.venv/lib/python3.*/site-packages/tensorrt_libs)
find_library(TRT_LIB NAMES libnvinfer.so.10 nvinfer HINTS ${TRT_LIB_DIR} ${TRT_VENV_DIRS} REQUIRED)
get_filename_component(TRT_LIB_DIR_RESOLVED ${TRT_LIB} DIRECTORY)

# ---- core: engine + task kernels. No video codecs, no drawing. -------------------------------
add_library(dfine SHARED
  src/core/api.cpp src/core/engine.cpp src/core/kernels_core.cu)
target_include_directories(dfine
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE src/core)
target_include_directories(dfine SYSTEM PUBLIC ${TRT_INCLUDE_DIR})
target_link_libraries(dfine PUBLIC ${TRT_LIB} CUDA::cudart)
target_compile_options(dfine PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>)
set_target_properties(dfine PROPERTIES
  BUILD_RPATH "${TRT_LIB_DIR_RESOLVED};${CUDAToolkit_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN"
  PUBLIC_HEADER include/dfine.h)

# ---- video: NVDEC -> core -> overlay -> NVENC. NVENC is absent on some Jetsons (Orin Nano),
# which is why this is a separate, optional target rather than the only way to use the model.
if(DFINE_BUILD_VIDEO)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(AV REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)
  add_executable(dfine_e2e
    src/video/main.cpp src/video/nvdec.cpp src/video/nvenc.cpp src/video/kernels_video.cu)
  target_include_directories(dfine_e2e PRIVATE src/video src/core)
  target_include_directories(dfine_e2e SYSTEM PRIVATE third_party)
  target_link_libraries(dfine_e2e PRIVATE
    dfine CUDA::cudart CUDA::cuda_driver PkgConfig::AV Threads::Threads ${CMAKE_DL_LIBS})
  target_compile_options(dfine_e2e PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>)
  set_target_properties(dfine_e2e PROPERTIES
    BUILD_RPATH "${TRT_LIB_DIR_RESOLVED};${CUDAToolkit_LIBRARY_DIR}")
endif()

if(DFINE_BUILD_EXAMPLES)
  foreach(ex detect bench)
    add_executable(dfine_${ex} examples/${ex}.cpp)
    target_link_libraries(dfine_${ex} PRIVATE dfine Threads::Threads)
    set_target_properties(dfine_${ex} PROPERTIES
      BUILD_RPATH "${TRT_LIB_DIR_RESOLVED};${CUDAToolkit_LIBRARY_DIR}")
  endforeach()
  set_target_properties(dfine_detect PROPERTIES OUTPUT_NAME dfine_example)
endif()

include(GNUInstallDirs)
install(TARGETS dfine
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
