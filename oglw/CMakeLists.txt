# oglw
add_library(oglw INTERFACE)
target_include_directories(oglw INTERFACE "include/")
target_link_libraries(oglw INTERFACE
    OpenGL::GL
    glad
    glm
    stb_image
    spdlog::spdlog
)