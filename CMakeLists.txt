cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
# Explicitly set architectures (or use "native")
set(CMAKE_CUDA_ARCHITECTURES 75 80 90)
set(TORCH_CUDA_ARCH_LIST "7.5 8.0 9.0")

# print env variable CUDACXX
message(STATUS "CUDACXX: $ENV{CUDACXX}")

project(HelloWorld VERSION 0.0.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# print nvcc path
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Toolkit bin directory: ${CUDAToolkit_BIN_DIR}")
message(STATUS "NVCC: ${CUDAToolkit_BIN_DIR}/nvcc")

# find CUDA Toolkit (system level cuda runtime)
find_package(CUDAToolkit REQUIRED)

# find cutlass and eigen
set(CUTLASS_DIR "cutlass")
set(EIGEN_DIR "eigen")

# Include header-only libraries
add_library(cutlass INTERFACE)
target_include_directories(cutlass INTERFACE
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${EIGEN_DIR})


# Find LibTorch
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Automatically find all main*.cu and main*.cpp files and create executables
file(GLOB MAIN_SOURCES "main*.cu" "main*.cpp")
list(APPEND MAIN_SOURCES "sgemm_sm80.cu")
foreach(MAIN_FILE ${MAIN_SOURCES})
    get_filename_component(EXEC_NAME ${MAIN_FILE} NAME_WE)
    add_executable(${EXEC_NAME} ${MAIN_FILE})
    target_link_libraries(${EXEC_NAME} PRIVATE "${TORCH_LIBRARIES}" cutlass eigen)
    message(STATUS "Added executable: ${EXEC_NAME} from ${MAIN_FILE}")
endforeach()

