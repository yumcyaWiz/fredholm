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

static void glfw_error_callback(int error, const char* description)
{
    spdlog::error("Glfw Error %d: %s\n", error, description);
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
    GLFWwindow* window =
        glfwCreateWindow(512, 512, "fredholm", nullptr, nullptr);
    if (!window) { return EXIT_FAILURE; }
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // init glad
    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress))
    {
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

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // start imgui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

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

    return 0;
}