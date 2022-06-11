# sutil
add_library(sutil INTERFACE)
target_include_directories(sutil INTERFACE "sutil/")

# tinyobjloader
add_library(tinyobjloader INTERFACE)
target_include_directories(tinyobjloader INTERFACE "tinyobjloader/")

# glm
add_subdirectory("glm/")