# OpenGL
find_package(OpenGL REQUIRED)

# viewer
add_executable(viewer "main.cpp")
target_compile_features(viewer PUBLIC cxx_std_20)
set_target_properties(viewer PROPERTIES CXX_EXTENSIONS OFF)

target_link_libraries(viewer PRIVATE spdlog::spdlog)

target_link_libraries(viewer PRIVATE OpenGL::GL)
target_link_libraries(viewer PRIVATE glad)
target_link_libraries(viewer PRIVATE glfw)
target_link_libraries(viewer PRIVATE imgui)
target_link_libraries(viewer PRIVATE imgui_glfw_opengl3)