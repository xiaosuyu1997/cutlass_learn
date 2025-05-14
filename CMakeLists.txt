cmake_minimum_required(VERSION 3.20)
# Explicitly set architectures (or use "native")
set(CMAKE_CUDA_ARCHITECTURES 75 80)

project(HelloWorld VERSION 0.0.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# print nvcc path
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Toolkit bin directory: ${CUDAToolkit_BIN_DIR}")
message(STATUS "NVCC: ${CUDAToolkit_BIN_DIR}/nvcc")

# find CUDA Toolkit (system level cuda runtime)
find_package(CUDAToolkit REQUIRED)

# find cutlass
set(CUTLASS_DIR "cutlass")

include_directories(
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

add_executable(HelloWorld main.cu)
add_executable(sgemm sgemm_sm80.cu)
