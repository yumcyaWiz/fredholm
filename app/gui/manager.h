#pragma once
#include <filesystem>
#include <string>
#include <vector>

#include "imgui.h"
#include "loader.h"
#include "renderer.h"

namespace fredholm
{

inline float deg_to_rad(float deg) { return deg / 180.0f * M_PI; }
inline float rad_to_deg(float rad) { return rad / M_PI * 180.0f; }

struct SceneListEntry
{
    std::string name;
    std::filesystem::path filepath;

    bool is_valid() const { return !filepath.empty(); }
};

struct EnvmapListEntry
{
    std::string name;
    std::filesystem::path filepath;

    bool is_valid() const { return !filepath.empty(); }
};

// https://qiita.com/syoyo/items/f6c219f243c3527f6121
static bool ImGuiComboUI(const std::string& caption, int current_idx,
                         const std::vector<std::string>& items)
{
    bool changed = false;

    if (ImGui::BeginCombo(caption.c_str(), items[current_idx].c_str()))
    {
        for (int n = 0; n < items.size(); n++)
        {
            bool is_selected = (current_idx == n);
            if (ImGui::Selectable(items[n].c_str(), is_selected))
            {
                current_idx = n;
                changed = true;
            }
            if (is_selected)
            {
                // Set the initial focus when opening the combo (scrolling + for
                // keyboard navigation support in the upcoming navigation
                // branch)
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    return changed;
}

class SceneManager
{
   public:
    SceneManager(OptixDeviceContext context) : context(context)
    {
        load_scene();
    }

    ~SceneManager()
    {
        if (scene_device) { scene_device.reset(); }
    }

    fredholm::Camera& get_camera() { return camera; }

    fredholm::DirectionalLight& get_directional_light()
    {
        return directional_light;
    }

    void run_imgui(fredholm::Renderer& renderer)
    {
        ImGui::Checkbox("animation", &animation);
        if (animation)
        {
            ImGui::InputFloat("animation time", &animation_time);
            update_animation(renderer);
        }

        // TODO: place these inside camera?
        const glm::vec3 origin = camera.get_origin();
        ImGui::Text("origin: (%f, %f, %f)", origin.x, origin.y, origin.z);
        const glm::vec3 forward = camera.get_forward();
        ImGui::Text("forward: (%f, %f, %f)", forward.x, forward.y, forward.z);

        camera_fov = rad_to_deg(camera.get_fov());
        if (ImGui::InputFloat("fov", &camera_fov))
        {
            camera.set_fov(deg_to_rad(camera_fov));
            renderer.clear_render();
        }

        camera_F = camera.get_F();
        if (ImGui::InputFloat("F", &camera_F))
        {
            camera.set_F(camera_F);
            renderer.clear_render();
        }

        camera_focus = camera.get_focus();
        if (ImGui::InputFloat("focus", &camera_focus))
        {
            camera.set_focus(camera_focus);
            renderer.clear_render();
        }

        camera_movement_speed = camera.get_movement_speed();
        if (ImGui::InputFloat("movement speed", &camera_movement_speed))
        {
            camera.set_movement_speed(camera_movement_speed);
        }

        const std::string scene_list = get_scene_list_for_imgui();
        const std::string envmap_list = get_envmap_list_for_imgui();

        // if CollapsingHeader and Combo box has the same name, the combo box
        // doesn't work
        if (ImGui::Combo("Scene", &m_scene_index, scene_list.c_str()))
        {
            load_scene();
            renderer.clear_render();
        }

        if (ImGui::Checkbox("Use arhosek sky", &use_arhosek))
        {
            load_arhosek();
            renderer.clear_render();
        }

        if (use_arhosek)
        {
            if (ImGui::InputFloat("intensity", &arhosek_intensity))
            {
                load_arhosek();
                renderer.clear_render();
            }
            if (ImGui::InputFloat("turbidity", &arhosek_turbidity))
            {
                load_arhosek();
                renderer.clear_render();
            }
            if (ImGui::InputFloat("albedo", &arhosek_albedo))
            {
                load_arhosek();
                renderer.clear_render();
            }
        }
        else
        {
            if (ImGui::Combo("Envmap", &m_envmap_index, envmap_list.c_str()))
            {
                load_envmap();
                renderer.clear_render();
            }
        }

        // TODO: sync parameters
        if (ImGui::InputFloat3("directional light color",
                               directional_light_color))
        {
            directional_light.le = make_float3(directional_light_color[0],
                                               directional_light_color[1],
                                               directional_light_color[2]);
            renderer.clear_render();
        }

        if (ImGui::InputFloat3("directional light direction",
                               directional_light_direction))
        {
            directional_light.dir = normalize(make_float3(
                directional_light_direction[0], directional_light_direction[1],
                directional_light_direction[2]));
            renderer.clear_render();

            if (use_arhosek)
            {
                load_arhosek();
                renderer.clear_render();
            }
        }

        if (ImGui::InputFloat("directional light angle",
                              &directional_light_angle))
        {
            directional_light.angle = deg_to_rad(directional_light_angle);
            renderer.clear_render();
        }

        ImGui::Text("# of vertices: %d", scene_device->get_n_vertices());
        ImGui::Text("# of faces: %d", scene_device->get_n_faces());
        ImGui::Text("# of materials: %d", scene_device->get_n_materials());
        ImGui::Text("# of textures: %d", scene_device->get_n_textures());
        ImGui::Text("# of geometries: %d", scene_device->get_n_geometries());
        ImGui::Text("# of instances: %d", scene_device->get_n_instances());
        ImGui::Text("# of area lights: %d", scene_device->get_n_area_lights());
    }

    const SceneDevice& get_scene_device() const { return *scene_device; }

   private:
    void update_animation(fredholm::Renderer& renderer)
    {
        animation_time += ImGui::GetIO().DeltaTime;

        scene_graph.update_animation(animation_time);
        fredholm::CompiledScene compiled_scene = scene_graph.compile();
        // TODO: update object transforms
        camera = compiled_scene.camera;

        renderer.clear_render();
    }

    void load_scene()
    {
        const auto& entry = m_scenes[m_scene_index];
        if (entry.is_valid())
        {
            fredholm::SceneLoader::load(entry.filepath, scene_graph);
            const auto& envmap_entry = m_envmaps[m_envmap_index];
            fredholm::SceneLoader::load_envmap(envmap_entry.filepath,
                                               scene_graph);
            fredholm::CompiledScene compiled_scene = scene_graph.compile();

            camera = compiled_scene.camera;

            scene_device = std::make_unique<fredholm::SceneDevice>();
            scene_device->send(context, compiled_scene);
        }
    }

    void load_envmap()
    {
        const auto& entry = m_envmaps[m_envmap_index];
        if (entry.is_valid())
        {
            fredholm::SceneLoader::load_envmap(entry.filepath, scene_graph);
            // TODO: update only envmap
            fredholm::CompiledScene compiled_scene = scene_graph.compile();

            scene_device = std::make_unique<fredholm::SceneDevice>();
            scene_device->send(context, compiled_scene);
        }
    }

    void load_arhosek()
    {
        const float3 sun_direction = normalize(make_float3(
            directional_light_direction[0], directional_light_direction[1],
            directional_light_direction[2]));
        scene_device->update_arhosek(arhosek_intensity, sun_direction,
                                     arhosek_turbidity, arhosek_albedo);
    }

    std::string get_scene_list_for_imgui() const
    {
        std::string names;
        for (const auto& entry : m_scenes) { names += entry.name + '\0'; }
        names += '\0';
        return names;
    }

    std::string get_envmap_list_for_imgui() const
    {
        std::string names;
        for (const auto& entry : m_envmaps) { names += entry.name + '\0'; }
        names += '\0';
        return names;
    }

    OptixDeviceContext context = nullptr;
    fredholm::SceneGraph scene_graph = {};
    fredholm::Camera camera = {};
    fredholm::DirectionalLight directional_light = {};
    std::unique_ptr<fredholm::SceneDevice> scene_device = nullptr;

    // for imgui
    bool animation = false;
    float animation_time = 0.0f;

    float camera_fov = 90.0f;
    float camera_F = 1.0f;
    float camera_focus = 1.0f;
    float camera_movement_speed = 1.0f;

    float directional_light_color[3] = {0.0f, 0.0f, 0.0f};
    float directional_light_direction[3] = {0.0f, 0.0f, 0.0f};
    float directional_light_angle = 0.0f;

    bool use_arhosek = false;
    float arhosek_intensity = 1.0f;
    float arhosek_turbidity = 1.0f;
    float arhosek_albedo = 0.3f;

    int m_scene_index = 0;
    int m_envmap_index = 1;

    const std::vector<SceneListEntry> m_scenes = {
        {"CornellBox-Original",
         std::filesystem::path(CMAKE_SOURCE_DIR) /
             "resources/scenes/cornellbox/CornellBox-Original.obj"},
        {"CornellBox-Texture",
         std::filesystem::path(CMAKE_SOURCE_DIR) /
             "resources/scenes/cornellbox/CornellBox-Texture.obj"},
        {"CornellBox(gltf)", std::filesystem::path(CMAKE_SOURCE_DIR) /
                                 "resources/scenes/cornellbox/CornellBox.gltf"},
        {"CornellBox-Textured(gltf)",
         std::filesystem::path(CMAKE_SOURCE_DIR) /
             "resources/scenes/cornellbox/CornellBox-Textured.gltf"},
        {"CornellBox-Specular(gltf)",
         std::filesystem::path(CMAKE_SOURCE_DIR) /
             "resources/scenes/cornellbox/CornellBox-Specular.gltf"},
        {"CornellBox-Metal(gltf)",
         std::filesystem::path(CMAKE_SOURCE_DIR) /
             "resources/scenes/cornellbox/CornellBox-Metal.gltf"},
        {"CornellBox-Transmission(gltf)",
         std::filesystem::path(CMAKE_SOURCE_DIR) /
             "resources/scenes/cornellbox/CornellBox-Transmission.gltf"},
        {"CornellBox-Normal(gltf)",
         std::filesystem::path(CMAKE_SOURCE_DIR) /
             "resources/scenes/cornellbox/CornellBox-Normal.gltf"},
        {"CornellBox-Camera-Animated(gltf)",
         std::filesystem::path(CMAKE_SOURCE_DIR) /
             "resources/scenes/cornellbox/CornellBox-Camera-Animated.gltf"},
        {"test", std::filesystem::path(CMAKE_SOURCE_DIR) /
                     "resources/scenes/test/test.json"},
        {"Sponza", std::filesystem::path(CMAKE_SOURCE_DIR) /
                       "resources/scenes/sponza/Sponza.gltf"},
        {"Modern Villa",
         std::filesystem::path(CMAKE_SOURCE_DIR) /
             "resources/scenes/modern_villa/"
             "Modern villa 2021 Blender Eevee and Cycles 2 DAY.gltf"},
        {"Modern Mansion", std::filesystem::path(CMAKE_SOURCE_DIR) /
                               "resources/scenes/blender_eevee_modern_mansion/"
                               "modern_mansion_2_eevee.gltf"},
        {"Gothic Library",
         std::filesystem::path(CMAKE_SOURCE_DIR) /
             "resources/scenes/gothic_library/Gothic library 2 EEVEE.gltf"},
        {"AI58_009", std::filesystem::path(CMAKE_SOURCE_DIR) /
                         "resources/scenes/ai58/AI58_009.gltf"},
        {"AE33_006", std::filesystem::path(CMAKE_SOURCE_DIR) /
                         "resources/scenes/ae33_006/AE33_006.obj"},
    };

    const std::vector<EnvmapListEntry> m_envmaps = {
        {"Black", std::filesystem::path(CMAKE_SOURCE_DIR) /
                      "resources/envmap/black.hdr"},
        {"White", std::filesystem::path(CMAKE_SOURCE_DIR) /
                      "resources/envmap/white.hdr"},
        {"Uffizi", std::filesystem::path(CMAKE_SOURCE_DIR) /
                       "resources/envmap/uffizi-large.hdr"}};
};

}  // namespace fredholm