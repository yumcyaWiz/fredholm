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

# external
add_subdirectory("externals/")

# fredholm
add_subdirectory("fredholm/")

# app
add_subdirectory("app/")