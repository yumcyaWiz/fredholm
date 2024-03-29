# compile modules into PTX
add_library(modules OBJECT
    "modules/pt.cu"
)

set_target_properties(modules PROPERTIES
    CUDA_PTX_COMPILATION ON
)
target_compile_options(modules PRIVATE
    --optix-ir
)
target_include_directories(modules PRIVATE
    ${OptiX_INCLUDE}
    "include/"
)
target_link_libraries(modules PRIVATE
    sutil
)

# kernels
add_subdirectory("kernels/")

# fredholm
add_library(fredholm
    "src/renderer.cpp"
    "src/scene.cpp"
)
target_include_directories(fredholm PUBLIC
    "include/"
)
target_link_libraries(fredholm PUBLIC
    cwl
    optwl
    tinyobjloader
    tinygltf
    sutil
    glm
    stb_image
    spdlog::spdlog
    kernels
)

# set modules source dir macro
target_compile_definitions(fredholm PUBLIC MODULES_SOURCE_DIR="${CMAKE_BINARY_DIR}/fredholm/CMakeFiles/modules.dir/modules")