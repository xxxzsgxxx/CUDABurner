cmake_minimum_required(VERSION 3.18)

# To make the project portable, we let CMake find the CUDA compiler.
# If you have multiple CUDA versions installed, you can specify one by setting
# the CUDACXX environment variable or by passing -DCMAKE_CUDA_COMPILER=/path/to/nvcc
# to the cmake command.

# Set architectures BEFORE project() to guide the compiler check
# Compile for a wide range of modern architectures to ensure compatibility (fatbinary)
# 75: Turing, 80/86/89: Ampere/Ada, 90: Hopper
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 120)

project(CUDABurner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(Threads REQUIRED)

# CUDA include path - auto-detected or user-provided
# Usage: cmake .. -DCUDA_INCLUDE_DIR=/usr/local/cuda/include
if(DEFINED CUDA_INCLUDE_DIR)
    message(STATUS "Using CUDA include dir from user: ${CUDA_INCLUDE_DIR}")
elseif(CUDAToolkit_INCLUDE_DIRS)
    set(CUDA_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "Using CUDA include dir from CUDAToolkit: ${CUDA_INCLUDE_DIR}")
elseif(CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include")
    message(STATUS "Using CUDA include dir from CUDA_TOOLKIT_ROOT_DIR: ${CUDA_INCLUDE_DIR}")
else()
    find_path(CUDA_INCLUDE_DIR nvml.h
        PATHS /usr/local/cuda/include /opt/cuda/include
        NO_DEFAULT_PATH
    )
    if(CUDA_INCLUDE_DIR)
        message(STATUS "Auto-detected CUDA include dir: ${CUDA_INCLUDE_DIR}")
    else()
        message(WARNING "CUDA include directory not found. Set -DCUDA_INCLUDE_DIR=/path/to/cuda/include")
    endif()
endif()

if(CUDA_INCLUDE_DIR AND EXISTS "${CUDA_INCLUDE_DIR}/nvml.h")
    list(APPEND EXTRA_INCLUDE_DIRS "${CUDA_INCLUDE_DIR}")
endif()

find_library(NVML_LIB nvidia-ml DOC "Path to NVML library")
if(NOT NVML_LIB)
    message(FATAL_ERROR "NVML library (nvidia-ml) not found.")
endif()

add_executable(CUDABurner
    main.cu
    monitoring/monitor.cpp
    operators/matrix_size.cpp
    operators/gemm_generic.cu
    operators/gemm_tensorcore.cu
    operators/gemm_sparse_tensorcore.cu
    operators/gemm_fp8_tensorcore.cu
    operators/gemm_fp4_tensorcore.cu
    operators/gemm_int4_tensorcore.cu
    operators/operator_factory.cu
    strategies/stress_strategy.cpp
    strategies/benchmark_strategy.cpp
    ui/tui.cpp
)

target_include_directories(CUDABurner PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${EXTRA_INCLUDE_DIRS}
)

# Manually define CUDA_VERSION for the preprocessor as it's not being picked up automatically
# in operator_factory.cu. This is likely because the file contains no device code.
if(CMAKE_CUDA_COMPILER_VERSION)
    string(REGEX MATCH "^[0-9]+" CUDA_MAJOR "${CMAKE_CUDA_COMPILER_VERSION}")
    string(REGEX MATCH "\\.[0-9]+" CUDA_MINOR_WITH_DOT "${CMAKE_CUDA_COMPILER_VERSION}")
    string(SUBSTRING "${CUDA_MINOR_WITH_DOT}" 1 -1 CUDA_MINOR)
    
    math(EXPR CUDA_VERSION_INT "${CUDA_MAJOR} * 1000 + ${CUDA_MINOR} * 10")
    
    target_compile_definitions(CUDABurner PRIVATE "CUDA_VERSION=${CUDA_VERSION_INT}")
    message(STATUS "Manually setting CUDA_VERSION preprocessor definition to ${CUDA_VERSION_INT}")
else()
    message(WARNING "CMAKE_CUDA_COMPILER_VERSION is not available. CUDA_VERSION macro will not be set.")
endif()


# --- 终极釜底抽薪之计 ---
# 我们把所有库都链接到最终的可执行文件上
# 并且把 cudart 放在链接列表的最后，这有时能解决顺序问题
target_link_libraries(CUDABurner PRIVATE
    Threads::Threads
    ${NVML_LIB}
    cublas
    cublasLt
    cudart
)

target_compile_options(CUDABurner PRIVATE -Wall -Wextra -O3)
install(TARGETS CUDABurner DESTINATION bin)

# Install the .desktop file for AppImage/Linux integration
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/CUDABurner.desktop"
        DESTINATION share/applications)

# Install the icon file
# It's assumed the icon is generated before cmake is run, or is present in the source dir.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cudaburner.png"
        DESTINATION share/icons/hicolor/128x128/apps
        RENAME cudaburner.png)
