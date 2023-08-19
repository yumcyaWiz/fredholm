# optix kernel
add_library(optix_kernel OBJECT
    "render_strategy/hello/hello.cu"
    "render_strategy/simple/simple.cu"
    "render_strategy/pt/pt.cu"
)
set_target_properties(optix_kernel PROPERTIES
    CUDA_PTX_COMPILATION ON
)
target_compile_options(optix_kernel PRIVATE
    --optix-ir
)
target_include_directories(optix_kernel PRIVATE
    "./"
    ${OptiX_INCLUDE}
)

# fredholm
add_library(fredholm
    "fredholm.cpp"
)
target_include_directories(fredholm PUBLIC
    "./"
    ${OptiX_INCLUDE}
)
target_link_libraries(fredholm PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    glm
    glad
    stb_image
    stb_image_write
    tinygltf
    tinyobjloader
    spdlog::spdlog
    glsl-shader-includes
    imgui
)

# post process
add_subdirectory("post_process/")