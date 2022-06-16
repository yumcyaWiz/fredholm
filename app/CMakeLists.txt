# cui
# add_executable(cui "cui.cpp")
# target_link_libraries(cui PRIVATE fredholm)

# OpenGL
find_package(OpenGL REQUIRED)

# gcss
add_library(gcss INTERFACE)
target_include_directories(gcss INTERFACE "gcss/")
target_link_libraries(gcss INTERFACE
    OpenGL::GL
    glad
    glm
    stb
    spdlog::spdlog
)

# gui
add_executable(gui "gui.cpp")
target_link_libraries(gui PRIVATE
    spdlog::spdlog
    OpenGL::GL
    glad
    gcss
    glfw
    imgui
    imgui_glfw_opengl3
    CUDA::cudart
    fredholm
)

# set cmake source dir macro
target_compile_definitions(gui PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}" CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")