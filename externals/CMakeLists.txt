# tinyobjloader
add_library(tinyobjloader
    "tinyobjloader.cpp"
)
target_include_directories(tinyobjloader PUBLIC "tinyobjloader/")

# glm
add_library(glm INTERFACE)
target_include_directories(glm SYSTEM INTERFACE "glm/")

# glad
add_library(glad glad/src/gl.c)
target_include_directories(glad SYSTEM PUBLIC glad/include)

# glfw
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)

# imgui
add_library(imgui imgui/imgui.cpp imgui/imgui_widgets.cpp imgui/imgui_tables.cpp imgui/imgui_draw.cpp imgui/imgui_demo.cpp)
target_include_directories(imgui SYSTEM PUBLIC imgui)

# imgui backend
add_library(imgui_glfw_opengl3 imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp)
target_compile_definitions(imgui_glfw_opengl3 PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(imgui_glfw_opengl3 imgui)
target_link_libraries(imgui_glfw_opengl3 glad)
target_link_libraries(imgui_glfw_opengl3 glfw)
target_include_directories(imgui_glfw_opengl3 SYSTEM PUBLIC imgui/backends)

# spdlog
add_subdirectory("spdlog/")

# stb_image
add_library(stb_image
    "stb_image.cpp"
)
target_include_directories(stb_image SYSTEM PUBLIC "stb/")

# stb_image_write
add_library(stb_image_write
    "stb_image_write.cpp"
)
target_include_directories(stb_image_write SYSTEM PUBLIC "stb/")

# tinygltf
add_library(tinygltf
    "tinygltf.cpp"
)
target_include_directories(tinygltf SYSTEM PUBLIC "tinygltf/")
target_link_libraries(tinygltf PUBLIC
    stb_image
    stb_image_write
)

# argparse
add_library(argparse INTERFACE)
target_include_directories(argparse SYSTEM INTERFACE "argparse/include")

# GLSL-Shader-Includes
add_library(glsl-shader-includes INTERFACE)
target_include_directories(glsl-shader-includes SYSTEM INTERFACE GLSL-Shader-Includes)

# json
# Typically you don't care so much for a third party library's tests to be
# run from your own project's code.
set(JSON_BuildTests OFF CACHE INTERNAL "")

# If you only include this third party in PRIVATE source files, you do not
# need to install it when your main project gets installed.
# set(JSON_Install OFF CACHE INTERNAL "")

# Don't use include(nlohmann_json/CMakeLists.txt) since that carries with it
# unintended consequences that will break the build.  It's generally
# discouraged (although not necessarily well documented as such) to use
# include(...) for pulling in other CMake projects anyways.
add_subdirectory("json/")