#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "glad/gl.h"
//
#include "GLFW/glfw3.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "spdlog/spdlog.h"
//
#include "oglw/quad.h"
#include "oglw/shader.h"
//
#include "controller.h"

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
    controller.clear_render();
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    controller.move_camera(fredholm::CameraMovement::LEFT, io.DeltaTime);
    controller.clear_render();
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    controller.move_camera(fredholm::CameraMovement::BACKWARD, io.DeltaTime);
    controller.clear_render();
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    controller.move_camera(fredholm::CameraMovement::RIGHT, io.DeltaTime);
    controller.clear_render();
  }

  // camera look around
  if (!io.WantCaptureMouse &&
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
    controller.rotate_camera(io.MouseDelta.x, io.MouseDelta.y);
    controller.clear_render();
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

  {
    // prepare controller
    Controller controller;
    controller.init_camera();

    controller.init_renderer();
    controller.load_scene();
    controller.init_render_layers();
    controller.clear_render();

    controller.init_denoiser();

    // prepare quad
    oglw::Quad quad;

    // prepare shaders
    oglw::VertexShader vertex_shader(
        std::filesystem::path(CMAKE_CURRENT_SOURCE_DIR) / "shaders" /
        "quad.vert");
    oglw::FragmentShader fragment_shader(
        std::filesystem::path(CMAKE_CURRENT_SOURCE_DIR) / "shaders" /
        "quad.frag");

    // create render pipeline
    oglw::Pipeline render_pipeline;
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
        {
          if (ImGui::Combo(
                  "Scene", &controller.m_imgui_scene_id,
                  "CornellBox\0RT Camp 8\0Sponza\0PBR Sponza\0Modern "
                  "Sponza\0Modern Sponza Small\0Salle de "
                  "bain\0Sibenik\0San "
                  "Miguel\0Rungholt\0Vokselia\0BMW\0BMW glTF\0Specular "
                  "Test\0Specular Roughness Test\0Metal Test\0Coat "
                  "Test\0Transmission "
                  "Test\0Transmission Roughness Test\0Transmission "
                  "Test(Sphere)\0Specular "
                  "Transmission Test\0Specular Transmission Roughness "
                  "Test\0Sheen Test\0Diffuse Test\0Diffuse "
                  "Transmission Test\0Texture "
                  "Test\0Normal Map Test\0Specular White Furnace "
                  "Test\0Coat White Furnace Test\0Metal Rough Spheres\0Clear "
                  "Coat Test\0Emission Texture Test\0Instance "
                  "Test\0Animation Test\0Camera Animation "
                  "Test\0Box\0Cube\0Animated Cube\0glTF "
                  "Test\0glTF Test2\0\0")) {
            controller.load_scene();
          }

          if (ImGui::InputInt2("Resolution", controller.m_imgui_resolution)) {
            controller.update_resolution();
          }

          if (ImGui::InputInt("Max samples", &controller.m_imgui_max_samples)) {
            controller.clear_render();
          }
          if (ImGui::InputInt("Max depth", &controller.m_imgui_max_depth)) {
            controller.clear_render();
          }

          ImGui::Combo("AOV",
                       reinterpret_cast<int*>(&controller.m_imgui_aov_type),
                       "Beauty\0Denoised\0Position\0Normal\0Depth\0TexCoord\0Al"
                       "bedo\0\0");

          if (ImGui::InputFloat("time", &controller.m_imgui_time)) {
            controller.set_time();
            controller.clear_render();
          }

          ImGui::Checkbox("Play animation", &controller.m_imgui_play_animation);
          if (controller.m_imgui_play_animation) {
            ImGui::InputFloat("Time step", &controller.m_imgui_timestep);
          }

          ImGui::Text("spp: %d", controller.m_imgui_n_samples);

          ImGui::InputText("filename", controller.m_imgui_filename, 256);
          if (ImGui::Button("Save image")) { controller.save_image(); }
        }

        ImGui::Separator();

        {
          if (ImGui::InputFloat3("Origin", controller.m_imgui_origin)) {
            controller.update_camera();
            controller.clear_render();
          }
          if (ImGui::InputFloat("FOV", &controller.m_imgui_fov)) {
            controller.update_camera();
            controller.clear_render();
          }
          if (ImGui::InputFloat("F number", &controller.m_imgui_F)) {
            controller.update_camera();
            controller.clear_render();
          }
          if (ImGui::InputFloat("Focus distance", &controller.m_imgui_focus)) {
            controller.update_camera();
            controller.clear_render();
          }
          if (ImGui::InputFloat("Movement speed",
                                &controller.m_imgui_movement_speed)) {
            controller.update_camera();
          }
          if (ImGui::InputFloat("Rotation speed",
                                &controller.m_imgui_rotation_speed)) {
            controller.update_camera();
          }
        }

        ImGui::Separator();

        {
          if (ImGui::InputFloat3("Directional light emission",
                                 controller.m_imgui_directional_light_le)) {
            controller.update_directional_light();
            controller.clear_render();
          }
          if (ImGui::InputFloat3("Directional light direction",
                                 controller.m_imgui_directional_light_dir)) {
            controller.update_directional_light();

            // sync arhosek sky
            if (controller.m_imgui_sky_type == SkyType::ARHOSEK) {
              controller.load_arhosek();
            }

            controller.clear_render();
          }
          if (ImGui::InputFloat("Directional light angle",
                                &controller.m_imgui_directional_light_angle)) {
            controller.update_directional_light();
            controller.clear_render();
          }
        }

        ImGui::Separator();

        {
          if (ImGui::Combo("Sky Type",
                           reinterpret_cast<int*>(&controller.m_imgui_sky_type),
                           "Constant\0IBL\0Arhosek\0\0")) {
            controller.update_sky_type();
            controller.clear_render();
          }

          switch (controller.m_imgui_sky_type) {
            case SkyType::CONSTANT: {
              if (ImGui::InputFloat3("Background color",
                                     controller.m_imgui_bg_color)) {
                controller.clear_render();
              }
            } break;
            case SkyType::IBL: {
              if (ImGui::Combo("IBL", &controller.m_imgui_ibl_id,
                               "PaperMill3E\0\0")) {
                controller.load_ibl();
                controller.clear_render();
              }

              if (ImGui::InputFloat("Intensity",
                                    &controller.m_imgui_sky_intensity)) {
                controller.set_sky_intensity();
                controller.clear_render();
              }

            } break;
            case SkyType::ARHOSEK: {
              if (ImGui::InputFloat("Turbidity",
                                    &controller.m_imgui_arhosek_turbidity)) {
                controller.load_arhosek();
                controller.clear_render();
              }

              if (ImGui::InputFloat("Albedo",
                                    &controller.m_imgui_arhosek_albedo)) {
                controller.load_arhosek();
                controller.clear_render();
              }

              if (ImGui::InputFloat("Intensity",
                                    &controller.m_imgui_sky_intensity)) {
                controller.set_sky_intensity();
                controller.clear_render();
              }
            } break;
          }
        }

        ImGui::Separator();

        {
          ImGui::Checkbox("Use bloom", &controller.m_imgui_use_bloom);
          if (controller.m_imgui_use_bloom) {
            ImGui::InputFloat("Bloom threshold",
                              &controller.m_imgui_bloom_threshold);
            ImGui::InputFloat("Bloom sigma", &controller.m_imgui_bloom_sigma);
          }

          ImGui::InputFloat("ISO", &controller.m_imgui_iso);
          ImGui::InputFloat("Chromatic aberration",
                            &controller.m_imgui_chromatic_aberration);
        }
      }
      ImGui::End();

      // render
      controller.render();

      // denoise
      if (controller.m_imgui_aov_type == AOVType::DENOISED) {
        controller.denoise();
      }

      // post process
      controller.post_process();

      // render AOVs
      glClear(GL_COLOR_BUFFER_BIT);
      glViewport(0, 0, controller.m_imgui_resolution[0],
                 controller.m_imgui_resolution[1]);
      fragment_shader.setUniform("resolution",
                                 glm::vec2(controller.m_imgui_resolution[0],
                                           controller.m_imgui_resolution[1]));
      fragment_shader.setUniform("aov_type",
                                 static_cast<int>(controller.m_imgui_aov_type));
      controller.m_layer_beauty_pp->get_gl_buffer().bindToShaderStorageBuffer(
          0);
      controller.m_layer_denoised_pp->get_gl_buffer().bindToShaderStorageBuffer(
          1);
      controller.m_layer_position->get_gl_buffer().bindToShaderStorageBuffer(2);
      controller.m_layer_normal->get_gl_buffer().bindToShaderStorageBuffer(3);
      controller.m_layer_depth->get_gl_buffer().bindToShaderStorageBuffer(4);
      controller.m_layer_texcoord->get_gl_buffer().bindToShaderStorageBuffer(5);
      controller.m_layer_albedo->get_gl_buffer().bindToShaderStorageBuffer(6);
      quad.draw(render_pipeline);

      // render imgui
      ImGui::Render();
      int display_w, display_h;
      glfwGetFramebufferSize(window, &display_w, &display_h);
      glViewport(0, 0, display_w, display_h);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      glfwSwapBuffers(window);
    }
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}