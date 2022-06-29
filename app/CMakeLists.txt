# cui
# add_executable(cui "cui.cpp")
# target_link_libraries(cui PRIVATE fredholm)

# gui
add_executable(gui "gui.cpp")
target_link_libraries(gui PRIVATE
    spdlog::spdlog
    OpenGL::GL
    glad
    oglw
    glfw
    imgui
    imgui_glfw_opengl3
    stb
    cwl
    optwl
    fredholm
)

# set cmake source dir macro
target_compile_definitions(gui PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}" CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")