cmake_minimum_required(VERSION 3.20)
project(fredholm LANGUAGES C CXX CUDA)

# C++ version and std
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# CUDA C++ version and std
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

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

# external
add_subdirectory("externals/")

# compile modules into PTX
add_library(modules OBJECT
    "modules/white.cu"
    "modules/triangle.cu"
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

# main
add_executable(main "src/main.cpp")
target_include_directories(main PRIVATE 
    ${OptiX_INCLUDE}
    "src/"
)
target_link_libraries(main PRIVATE 
    CUDA::cudart
    tinyobjloader
)

# set modules source dir macro
target_compile_definitions(main PRIVATE MODULES_SOURCE_DIR="${CMAKE_BINARY_DIR}/CMakeFiles/modules.dir/modules")