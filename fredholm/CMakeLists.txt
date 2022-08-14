# compile modules into PTX
add_library(modules OBJECT
    "modules/pt.cu"
)
set_target_properties(modules PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_PTX_COMPILATION ON
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
add_library(fredholm INTERFACE)
target_include_directories(fredholm INTERFACE
    "include/"
)
target_link_libraries(fredholm INTERFACE
    cwl
    tinyobjloader
    tinygltf
    sutil
    glm
    stb
    spdlog::spdlog
    kernels
)

# set modules source dir macro
target_compile_definitions(fredholm INTERFACE MODULES_SOURCE_DIR="${CMAKE_BINARY_DIR}/fredholm/CMakeFiles/modules.dir/modules")