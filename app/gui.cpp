#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "glad/gl.h"
//
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "GLFW/glfw3.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "renderer.h"
#include "spdlog/spdlog.h"
//
#include "gcss/texture.h"
#include "quad.h"
#include "shader.h"
//
#include "camera.h"
#include "controller.h"
#include "cuda_gl_util.h"
#include "device/util.h"
#include "scene.h"

static void glfw_error_callback(int error, const char* description)
{
  spdlog::error("Glfw Error %d: %s\n", error, description);
}

void handle_input(GLFWwindow* window, const ImGuiIO& io, Controller& controller)
{
  // close window
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }

  // move camera
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    controller.move_camera(fredholm::CameraMovement::FORWARD, io.DeltaTime);
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    controller.move_camera(fredholm::CameraMovement::LEFT, io.DeltaTime);
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    controller.move_camera(fredholm::CameraMovement::BACKWARD, io.DeltaTime);
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    controller.move_camera(fredholm::CameraMovement::RIGHT, io.DeltaTime);
  }

  // camera look around
  if (!io.WantCaptureMouse &&
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
    controller.rotate_camera(io.MouseDelta.x, io.MouseDelta.y);
  }
}

void framebuffer_size_callback([[maybe_unused]] GLFWwindow* window, int width,
                               int height)
{
  glViewport(0, 0, width, height);
}

int main()
{
  // init glfw
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) { return EXIT_FAILURE; }

  // init window and OpenGL context
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow* window = glfwCreateWindow(512, 512, "fredholm", nullptr, nullptr);
  if (!window) { return EXIT_FAILURE; }
  glfwMakeContextCurrent(window);

  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // init glad
  if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
    spdlog::error("failed to initialize OpenGL context");
    return EXIT_FAILURE;
  }

  // init imgui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;

  // set imgui style
  ImGui::StyleColorsDark();

  // init imgui backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 460 core");

  Controller controller;
  controller.init_renderer();
  controller.load_scene(std::filesystem::path(CMAKE_SOURCE_DIR) / "resources" /
                        "salle_de_bain/salle_de_bain.obj");
  controller.init_render_states();
  controller.init_framebuffer();

  CameraSettings camera_settings;
  camera_settings.origin = make_float3(0, 1, 5);
  camera_settings.forward = make_float3(0, 0, -1);
  camera_settings.fov = 0.5f * M_PI;
  controller.init_camera(camera_settings);

  // prepare quad
  gcss::Quad quad;

  // prepare shaders
  gcss::VertexShader vertex_shader(
      std::filesystem::path(CMAKE_CURRENT_SOURCE_DIR) / "shaders" /
      "quad.vert");
  gcss::FragmentShader fragment_shader(
      std::filesystem::path(CMAKE_CURRENT_SOURCE_DIR) / "shaders" /
      "quad.frag");

  // create render pipeline
  gcss::Pipeline render_pipeline;
  render_pipeline.attachVertexShader(vertex_shader);
  render_pipeline.attachFragmentShader(fragment_shader);

  // app loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    handle_input(window, io, controller);

    // start imgui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("UI");
    {
      if (ImGui::InputInt2("Resolution", controller.m_imgui_resolution)) {
        controller.update_resolution();
      }
    }
    ImGui::End();

    const uint32_t width = controller.get_width();
    const uint32_t height = controller.get_height();

    // render
    controller.render(1, 100);

    // render texture
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, width, height);
    fragment_shader.setUniform("resolution", glm::vec2(width, height));
    controller.get_gl_framebuffer().bindToShaderStorageBuffer(0);
    quad.draw(render_pipeline);

    // render imgui
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}