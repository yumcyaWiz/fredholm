#pragma once
#include <filesystem>
#include <string>
#include <vector>

#include "imgui.h"
#include "loader.h"
#include "renderer.h"

namespace fredholm
{

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

    void run_imgui(Renderer& renderer)
    {
        const std::string scene_list = get_scene_list_for_imgui();
        const std::string envmap_list = get_envmap_list_for_imgui();

        // if CollapsingHeader and Combo box has the same name, the combo box
        // doesn't work
        if (ImGui::Combo("Scene", &m_scene_index, scene_list.c_str()))
        {
            load_scene();
            renderer.clear_render();
        }
        if (ImGui::Combo("Envmap", &m_envmap_index, envmap_list.c_str()))
        {
            load_envmap();
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
    void load_scene()
    {
        const auto& entry = m_scenes[m_scene_index];
        if (entry.is_valid())
        {
            fredholm::SceneLoader::load(entry.filepath, scene_graph);
            scene_device = std::make_unique<fredholm::SceneDevice>();
            scene_device->send(context, scene_graph);
        }
    }

    void load_envmap()
    {
        const auto& entry = m_envmaps[m_envmap_index];
        if (entry.is_valid())
        {
            fredholm::SceneLoader::load_envmap(entry.filepath, scene_graph);
            // TODO: update only envmap
            scene_device = std::make_unique<fredholm::SceneDevice>();
            scene_device->send(context, scene_graph);
        }
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
    fredholm::SceneGraph scene_graph;
    std::unique_ptr<fredholm::SceneDevice> scene_device = nullptr;

    int m_scene_index = 0;
    int m_envmap_index = 0;

    const std::vector<SceneListEntry> m_scenes = {
        {"CornellBox-Original",
         std::filesystem::path(CMAKE_SOURCE_DIR) /
             "resources/scenes/cornellbox/CornellBox-Original.obj"},
        {"CornellBox-Texture",
         std::filesystem::path(CMAKE_SOURCE_DIR) /
             "resources/scenes/cornellbox/CornellBox-Texture.obj"},
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