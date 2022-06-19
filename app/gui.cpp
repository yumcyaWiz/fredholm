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
#include "cuda_gl_util.h"
#include "device/util.h"
#include "scene.h"

static void glfw_error_callback(int error, const char* description)
{
  spdlog::error("Glfw Error %d: %s\n", error, description);
}

void handle_input(GLFWwindow* window, const ImGuiIO& io,
                  fredholm::Camera& camera, fredholm::Renderer& renderer)
{
  // close window
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }

  // move camera
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    camera.move(fredholm::CameraMovement::FORWARD, io.DeltaTime);
    renderer.init_render_states();
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    camera.move(fredholm::CameraMovement::LEFT, io.DeltaTime);
    renderer.init_render_states();
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    camera.move(fredholm::CameraMovement::BACKWARD, io.DeltaTime);
    renderer.init_render_states();
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    camera.move(fredholm::CameraMovement::RIGHT, io.DeltaTime);
    renderer.init_render_states();
  }

  // camera look around
  if (!io.WantCaptureMouse &&
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
    camera.lookAround(io.MouseDelta.x, io.MouseDelta.y);
    renderer.init_render_states();
  }
}

void framebuffer_size_callback([[maybe_unused]] GLFWwindow* window, int width,
                               int height)
{
  glViewport(0, 0, width, height);
}

int main()
{
  const uint32_t width = 1024;
  const uint32_t height = 1024;

  // init glfw
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) { return EXIT_FAILURE; }

  // init window and OpenGL context
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow* window =
      glfwCreateWindow(width, height, "fredholm", nullptr, nullptr);
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

  // prepare framebuffer
  app::CUDAGLBuffer<float4> framebuffer(width, height);

  // setup renderer
#ifdef NDEBUG
  bool enable_validation_mode = false;
#else
  bool enable_validation_mode = true;
#endif

  fredholm::Renderer renderer(enable_validation_mode);
  renderer.create_context();
  renderer.create_module(std::filesystem::path(MODULES_SOURCE_DIR) / "pt.ptx");
  renderer.create_program_group();
  renderer.create_pipeline();

  fredholm::Scene scene;
  scene.load_obj(std::filesystem::path(CMAKE_SOURCE_DIR) / "resources" /
                 "salle_de_bain/salle_de_bain.obj");
  renderer.load_scene(scene);
  renderer.build_accel();
  renderer.create_sbt();

  renderer.set_resolution(width, height);
  renderer.init_render_states();

  const float3 cam_origin = make_float3(0.0f, 1.0f, 5.0f);
  const float3 cam_forward = make_float3(0.0f, 0.0f, -1.0f);
  fredholm::Camera camera(cam_origin, cam_forward);

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

    handle_input(window, io, camera, renderer);

    // start imgui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("UI");
    {
    }
    ImGui::End();

    // render
    renderer.render(camera, framebuffer.m_d_buffer, 1, 100);
    // TODO: Is is safe to remove this?
    renderer.wait_for_completion();

    // render texture
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, width, height);
    fragment_shader.setUniform("resolution", glm::vec2(width, height));
    framebuffer.m_buffer.bindToShaderStorageBuffer(0);
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