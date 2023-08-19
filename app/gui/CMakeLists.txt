# gui
add_executable(gui "main.cpp")
target_link_libraries(gui PRIVATE
    fredholm
    glad
    glfw
    imgui
    imgui_glfw_opengl3
    spdlog::spdlog_header_only
)

# set cmake source dir macro
target_compile_definitions(gui PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}" CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")