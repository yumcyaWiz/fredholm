# OptiX path
set(CMAKE_MODULE_PATH
    "/opt/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64/SDK/CMake/"
    ${CMAKE_MODULE_PATH}
)

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# OptiX
set(OptiX_INSTALL_DIR "/opt/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64")
find_package(OptiX REQUIRED)

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
    "src"
)
target_link_libraries(modules PRIVATE
    sutil
)

# fredholm
add_library(fredholm INTERFACE)
target_include_directories(fredholm INTERFACE
    ${OptiX_INCLUDE}
    "src/"
)
target_link_libraries(fredholm INTERFACE
    CUDA::cudart
    tinyobjloader
    sutil
)

# set modules source dir macro
target_compile_definitions(fredholm INTERFACE MODULES_SOURCE_DIR="${CMAKE_BINARY_DIR}/fredholm/CMakeFiles/modules.dir/modules")