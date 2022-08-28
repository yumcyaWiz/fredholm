add_library(kernels
    "src/post-process.cu"
)
target_include_directories(kernels PUBLIC
    "include/"
)
target_link_libraries(kernels PUBLIC
    sutil
    cwl
)