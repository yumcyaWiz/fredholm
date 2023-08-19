cmake_minimum_required(VERSION 3.27)
project(fredholm LANGUAGES C CXX CUDA)

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ version and std
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# CUDA C++ version and std
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# OptiX path
set(CMAKE_MODULE_PATH
    "/opt/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/CMake/"
    ${CMAKE_MODULE_PATH}
)

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# OptiX
set(OptiX_INSTALL_DIR "/opt/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64")
find_package(OptiX REQUIRED)

# OpenGL
find_package(OpenGL REQUIRED)

# external
add_subdirectory("externals/")

# optix kernel
add_library(optix_kernel OBJECT
    "src/render_strategy/hello/hello.cu"
    "src/render_strategy/simple/simple.cu"
    "src/render_strategy/pt/pt.cu"
)
set_target_properties(optix_kernel PROPERTIES
    CUDA_PTX_COMPILATION ON
)
target_compile_options(optix_kernel PRIVATE
    --optix-ir
)
target_include_directories(optix_kernel PRIVATE
    "src/"
    ${OptiX_INCLUDE}
)

# fredholm
add_library(fredholm
    "src/fredholm.cpp"
)
target_include_directories(fredholm PUBLIC
    "src/"
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
)

# app
add_subdirectory("app/")