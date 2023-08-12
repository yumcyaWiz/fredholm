# rtcamp8
# add_executable(rtcamp8
# "rtcamp8.cpp"
# )
# target_link_libraries(rtcamp8 PRIVATE
# fredholm
# kernels
# )

# controller
add_library(controller
    "controller.cpp"
)
target_link_libraries(controller PRIVATE
    spdlog::spdlog
    OpenGL::GL
    glad
    oglw
    glfw
    stb_image_write
    cwl
    optwl
    fredholm
    kernels
)

# gui
add_executable(gui
    "gui.cpp"
)
target_link_libraries(gui PRIVATE
    spdlog::spdlog
    oglw
    cwl
    optwl
    imgui
    imgui_glfw_opengl3
    fredholm
    controller
)

# set cmake source dir macro
target_compile_definitions(gui PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}" CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")