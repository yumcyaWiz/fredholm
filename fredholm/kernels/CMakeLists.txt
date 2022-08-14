add_library(kernels
    "src/post-process.cu"
)
set_target_properties(kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(kernels PUBLIC
    "include/"
)
target_link_libraries(kernels PUBLIC
    sutil
)