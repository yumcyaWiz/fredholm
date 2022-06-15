# cui
add_executable(cui "cui.cpp")
target_link_libraries(cui PRIVATE fredholm)

# OpenGL
find_package(OpenGL REQUIRED)

# gcss
add_library(gcss INTERFACE)
target_include_directories(gcss INTERFACE "gcss/")

# gui
add_executable(gui "gui.cpp")
target_link_libraries(gui PRIVATE spdlog::spdlog)
target_link_libraries(gui PRIVATE OpenGL::GL)
target_link_libraries(gui PRIVATE glad)
target_link_libraries(gui PRIVATE gcss)
target_link_libraries(gui PRIVATE glfw)
target_link_libraries(gui PRIVATE imgui)
target_link_libraries(gui PRIVATE imgui_glfw_opengl3)