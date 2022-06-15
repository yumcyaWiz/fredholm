#include <cstdlib>
#include <iostream>

#include "glad/gl.h"
//
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "GLFW/glfw3.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "spdlog/spdlog.h"
//
#include "gcss/texture.h"
#include "quad.h"
#include "shader.h"

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
  uint32_t width = 512;
  uint32_t height = 512;

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

  // prepare framebuffer texture
  gcss::Texture framebuffer(glm::uvec2(width, height), GL_RGBA32F, GL_RGBA,
                            GL_FLOAT);

  std::vector<float> data(4 * width * height);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      const int idx = 4 * i + 4 * width * j;
      data[idx + 0] = static_cast<float>(i) / width;
      data[idx + 1] = static_cast<float>(j) / height;
      data[idx + 2] = 1.0f;
      data[idx + 3] = 1.0f;
    }
  }
  framebuffer.setImage(data, glm::uvec2(width, height), GL_RGBA32F, GL_RGBA,
                       GL_FLOAT);

  // GL interop
  // cudaGraphicsResource* resource;
  // cudaGraphicsGLRegisterImage(&resource, framebuffer.getTextureName(),
  //                             GL_TEXTURE_2D,
  //                             cudaGraphicsRegisterFlagsWriteDiscard);
  // cudaGraphicsMapResources(1, &resource, 0);

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
    framebuffer.bindToTextureUnit(0);
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
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}