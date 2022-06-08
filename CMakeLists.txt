cmake_minimum_required(VERSION 3.20)
project(fredholm LANGUAGES C CXX CUDA)

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

# compile shaders into PTX
add_library(shaders OBJECT
    "shaders/white.cu"
)
set_target_properties(shaders PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_PTX_COMPILATION ON
)
target_include_directories(shaders PRIVATE ${OptiX_INCLUDE})

# main
add_executable(main "src/main.cpp")
target_compile_features(main PRIVATE cxx_std_20)
set_target_properties(main PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(main PRIVATE ${OptiX_INCLUDE})
target_link_libraries(main PRIVATE CUDA::cudart)