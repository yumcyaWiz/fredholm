cmake_minimum_required(VERSION 3.20)
project(fredholm LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

# fredholm
add_subdirectory("fredholm/")

# app
add_subdirectory("app/")