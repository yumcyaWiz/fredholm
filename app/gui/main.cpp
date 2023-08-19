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
#include "loader.h"
#include "render_strategy/pt/pt.h"
#include "renderer.h"

struct SceneListEntry
{
    std::string name;
    std::filesystem::path filepath;
};

class SceneList
{
   public:
    static std::string get_names_for_imgui()
    {
        std::string names;
        for (const auto& entry : m_entries) { names += entry.name + '\0'; }
        names += '\0';
        return names;
    }

    static SceneListEntry get_entry(uint32_t idx) { return m_entries[idx]; }

   private:
    static std::vector<SceneListEntry> m_entries;
};

std::vector<SceneListEntry> SceneList::m_entries = {
    {"CornellBox-Original",
     std::filesystem::path(CMAKE_SOURCE_DIR) /
         "resources/scenes/cornellbox/CornellBox-Original.obj"},
    {"CornellBox-Texture",
     std::filesystem::path(CMAKE_SOURCE_DIR) /
         "resources/scenes/cornellbox/CornellBox-Texture.obj"},
};

void glfw_error_callback(int error, const char* description)
{
    spdlog::error("Glfw Error %d: %s\n", error, description);
}

void framebuffer_size_callback([[maybe_unused]] GLFWwindow* window, int width,
                               int height)
{
    glViewport(0, 0, width, height);
}

void gl_debug_message_callback(GLenum source, GLenum type, GLuint id,
                               GLenum severity, GLsizei length,
                               const GLchar* message, const void* user_param)
{
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:
            spdlog::critical("[gl] {}", message);
            break;
        case GL_DEBUG_SEVERITY_MEDIUM:
            spdlog::error("[gl] {}", message);
            break;
        case GL_DEBUG_SEVERITY_LOW:
            spdlog::warn("[gl] {}", message);
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            spdlog::info("[gl] {}", message);
            break;
    }
}

class App
{
   public:
    App()
    {
#ifndef NDEBUG
        debug = true;
#else
        debug = false;
#endif

        init_glfw();
        init_glad();
        init_gl();
        init_imgui();

        init_cuda();
        init_optix();
        init_renderer();
        init_shaders();

        pipeline->loadVertexShader(
            std::filesystem::path(CMAKE_CURRENT_SOURCE_DIR) /
            "shaders/shader.vert");
        pipeline->loadFragmentShader(
            std::filesystem::path(CMAKE_CURRENT_SOURCE_DIR) /
            "shaders/shader.frag");
    }

    ~App() { release(); }

    void run()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            // start imgui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            run_imgui();

            glClear(GL_COLOR_BUFFER_BIT);

            if (renderer)
            {
                // render
                renderer->render(camera, *scene_device);
                renderer->synchronize();

                // show image
                const uint32_t width = renderer->get_option<uint32_t>("width");
                const uint32_t height =
                    renderer->get_option<uint32_t>("height");
                const fredholm::GLBuffer& beauty =
                    renderer->get_aov("beauty").get_gl_buffer();

                pipeline->setUniform("resolution", glm::vec2(width, height));
                beauty.bindToShaderStorageBuffer(0);

                glViewport(0, 0, width, height);
                quad->draw(*pipeline);
            }

            // render imgui
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
        }
    };

   private:
    void init_cuda()
    {
        fredholm::cuda_check(cuInit(0));
        device = std::make_unique<fredholm::CUDADevice>(0);
    }

    void init_optix()
    {
        optixInit();
        context = fredholm::optix_create_context(device->get_context(), debug);
    }

    void init_renderer()
    {
        renderer = std::make_unique<fredholm::Renderer>(context, debug);

        camera = fredholm::Camera(glm::vec3(0, 1, 2));

        fredholm::SceneLoader::load("CornellBox-Texture.obj", scene);
        scene_device = std::make_unique<fredholm::SceneDevice>();
        scene_device->send(context, scene);

        options.use_gl_interop = true;
        renderer->set_render_strategy(fredholm::RenderStrategyType::PT,
                                      options);
    }

    void init_glad()
    {
        if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress))
        {
            throw std::runtime_error("failed to initialize OpenGL context");
        }
    }

    void init_glfw()
    {
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit()) { std::runtime_error("failed to initialize GLFW"); }

        // init window and OpenGL context
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(512, 512, "fredholm", nullptr, nullptr);
        if (!window) { std::runtime_error("failed to create window"); }
        glfwMakeContextCurrent(window);

        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    }

    void init_gl()
    {
        if (debug)
        {
            glEnable(GL_DEBUG_OUTPUT);
            glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            glDebugMessageCallback(gl_debug_message_callback, nullptr);
        }
    }

    void init_imgui()
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        (void)io;

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 460 core");
    }

    void init_shaders()
    {
        pipeline = std::make_unique<fredholm::GLPipeline>();
        quad = std::make_unique<fredholm::GLQuad>();
    }

    void run_imgui()
    {
        ImGui::Begin("fredholm");
        {
            if (ImGui::CollapsingHeader("Camera settings"))
            {
                // TODO: show camera settings
            }

            if (ImGui::CollapsingHeader("Scene settings"))
            {
                // TODO: show scene settings

                const std::string scenes_names =
                    SceneList::get_names_for_imgui();

                static int selected_scene = 0;
                if (ImGui::Combo("Scene", &selected_scene,
                                 scenes_names.c_str()))
                {
                    fredholm::SceneLoader::load(
                        SceneList::get_entry(selected_scene).filepath, scene);
                    scene_device->send(context, scene);
                    renderer->clear_render();
                }
            }

            if (ImGui::CollapsingHeader(("Render settings")))
            {
                static int selected_render_strategy =
                    static_cast<int>(renderer->get_render_strategy_type());
                if (ImGui::Combo("Render strategy", &selected_render_strategy,
                                 "Hello\0Simple\0PT\0\0"))
                {
                    renderer->set_render_strategy(
                        fredholm::RenderStrategyType(selected_render_strategy),
                        options);
                }

                renderer->runImGui();
            }
        }
        ImGui::End();
    }

    void release()
    {
        if (pipeline) { pipeline.reset(); }
        if (quad) { quad.reset(); }

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();

        if (scene_device) { scene_device.reset(); }
        if (renderer) { renderer.reset(); }

        fredholm::optix_check(optixDeviceContextDestroy(context));

        if (device) { device.reset(); }
    }

    GLFWwindow* window = nullptr;
    bool debug = false;

    std::unique_ptr<fredholm::CUDADevice> device = nullptr;
    OptixDeviceContext context = nullptr;
    fredholm::Camera camera;
    fredholm::SceneGraph scene;
    std::unique_ptr<fredholm::SceneDevice> scene_device = nullptr;
    fredholm::RenderOptions options;
    std::unique_ptr<fredholm::Renderer> renderer = nullptr;

    std::unique_ptr<fredholm::GLPipeline> pipeline = nullptr;
    std::unique_ptr<fredholm::GLQuad> quad = nullptr;
};

int main()
{
    App app;

    app.run();

    return 0;
}