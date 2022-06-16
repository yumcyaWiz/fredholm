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
#include "device/util.h"
#include "scene.h"

static void glfw_error_callback(int error, const char* description)
{
  spdlog::error("Glfw Error %d: %s\n", error, description);
}

void handle_input(GLFWwindow* window, const ImGuiIO& io)
{
  // close window
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

void framebuffer_size_callback([[maybe_unused]] GLFWwindow* window, int width,
                               int height)
{
  glViewport(0, 0, width, height);
}

int main()
{
  const uint32_t width = 512;
  const uint32_t height = 512;

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
  gcss::Buffer framebuffer;
  std::vector<glm::vec4> data(width * height);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      data[i + width * j] =
          glm::vec4(static_cast<float>(i) / width,
                    static_cast<float>(j) / height, 1.0f, 1.0f);
    }
  }

  framebuffer.setData(data, GL_STATIC_DRAW);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  // get cuda device ptr from OpenGL texture
  struct cudaGraphicsResource* resource;
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
      &resource, framebuffer.getName(), cudaGraphicsRegisterFlagsWriteDiscard));
  CUDA_CHECK(cudaGraphicsMapResources(1, &resource));

  float4* d_framebuffer;
  size_t d_framebuffer_size;
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&d_framebuffer), &d_framebuffer_size, resource));

  // setup renderer
#ifdef NDEBUG
  bool enable_validation_mode = false;
#else
  bool enable_validation_mode = true;
#endif

  fredholm::Renderer renderer(width, height, enable_validation_mode);
  renderer.create_context();
  renderer.create_module(std::filesystem::path(MODULES_SOURCE_DIR) / "pt.ptx");
  renderer.create_program_group();
  renderer.create_pipeline();

  fredholm::Scene scene;
  scene.load_obj(std::filesystem::path(CMAKE_SOURCE_DIR) / "resources" /
                 "CornellBox-Original.obj");
  renderer.load_scene(scene);
  renderer.build_accel();
  renderer.create_sbt(scene);

  const float3 cam_origin = make_float3(0.0f, 1.0f, 3.0f);
  const float3 cam_forward = make_float3(0.0f, 0.0f, -1.0f);
  fredholm::Camera camera(cam_origin, cam_forward);

  renderer.render_to_framebuffer(camera, d_framebuffer, 1, 100);

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

    handle_input(window, io);

    // start imgui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("UI");
    {
    }
    ImGui::End();

    // render texture
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, width, height);
    fragment_shader.setUniform("resolution", glm::vec2(width, height));
    framebuffer.bindToShaderStorageBuffer(0);
    quad.draw(render_pipeline);

    // render imgui
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  // cleanup
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource));
  CUDA_CHECK(cudaGraphicsUnregisterResource(resource));

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}