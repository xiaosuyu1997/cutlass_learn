# Define the project
project(fa_test_project)

# -------------------------------------------------------------------------
# 1. flash attention kernels, including tridao's and custom
# -------------------------------------------------------------------------
function(add_fa_kernel_lib LIB_NAME DIR_NAME)
    file(GLOB_RECURSE LIB_SOURCES "src/${DIR_NAME}/*.cpp" "src/${DIR_NAME}/*.cu")

    add_library(${LIB_NAME} ${LIB_SOURCES})
    target_include_directories(${LIB_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/${DIR_NAME}
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(${LIB_NAME}
        PUBLIC "${TORCH_LIBRARIES}"
        PRIVATE cutlass
    )
endfunction()

# add_fa_kernel_lib(custom_fa_lib custom_fa)
add_fa_kernel_lib(tridao_fa_lib tridao_fa)

# -------------------------------------------------------------------------
# 2. fa_test_lib: Test utilities from test/test_fa/
# -------------------------------------------------------------------------
file(GLOB_RECURSE TEST_UTILS_SOURCES "test/test_fa/*.cpp" "test/test_fa/*.cu")
add_library(fa_test_lib ${TEST_UTILS_SOURCES})

target_include_directories(fa_test_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_fa
)

target_link_libraries(fa_test_lib
    PUBLIC
        tridao_fa_lib
        "${TORCH_LIBRARIES}"
    PRIVATE
        cutlass
)

# -------------------------------------------------------------------------
# 3. Executables: Build main files
# -------------------------------------------------------------------------
file(GLOB TEST_MAINS "test/main*.cu")

foreach(MAIN_FILE ${TEST_MAINS})
    get_filename_component(EXEC_NAME ${MAIN_FILE} NAME_WE)
    add_executable(fa_test_${EXEC_NAME} ${MAIN_FILE})

    target_link_libraries(fa_test_${EXEC_NAME} PRIVATE
        fa_test_lib
    )

    # Inherit compile features (C++17)
    # target_compile_features(fa_test_${EXEC_NAME} PRIVATE cxx_std_17)

    message(STATUS "Configured test executable: fa_test_${EXEC_NAME}")
endforeach()
