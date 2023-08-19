add_library(post_process_kernel OBJECT
    "tone_mapping_kernel.cu"
)
set_target_properties(post_process_kernel PROPERTIES
    CUDA_PTX_COMPILATION ON
)
target_link_libraries(post_process_kernel PUBLIC
    fredholm
)

add_library(post_process
    "post_process.cpp"
)
target_link_libraries(post_process PUBLIC
    CUDA::cuda_driver
    fredholm
)

# set cmake source dir macro
target_compile_definitions(post_process PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}" CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")