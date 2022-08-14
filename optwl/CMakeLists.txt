add_library(optwl
    "src/optwl.cpp"
)
target_include_directories(optwl PUBLIC
    ${OptiX_INCLUDE}
    "include/"
)
target_link_libraries(optwl PUBLIC
    CUDA::cudart
    spdlog::spdlog
)