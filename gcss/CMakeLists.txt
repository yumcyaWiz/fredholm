# gcss
add_library(gcss INTERFACE)
target_include_directories(gcss INTERFACE "include/")
target_link_libraries(gcss INTERFACE
    OpenGL::GL
    glad
    glm
    stb
    spdlog::spdlog
)