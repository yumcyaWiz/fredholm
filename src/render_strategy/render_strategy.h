#pragma once
#include "camera.h"
#include "helper_math.h"
#include "optix_util.h"
#include "scene.h"
#include "shared.h"
#include "types.h"
#include "util.h"

namespace fredholm
{

// TODO: use observer pattern to detect changes in RenderOptions
// TODO: separate render layers
class RenderStrategy
{
   public:
    RenderStrategy(const OptixDeviceContext& context, bool debug)
        : context(context), debug(debug)
    {
    }

    virtual ~RenderStrategy()
    {
        cuda_check(cuMemFree(sbt_record_set.raygen_records));
        cuda_check(cuMemFree(sbt_record_set.miss_records));
        cuda_check(cuMemFree(sbt_record_set.hitgroup_records));

        if (m_pipeline != nullptr)
        {
            optix_check(optixPipelineDestroy(m_pipeline));
            m_pipeline = nullptr;
        }

        for (auto& raygen_program_group :
             m_program_group_sets.raygen_program_groups)
        {
            if (raygen_program_group != nullptr)
            {
                optix_check(optixProgramGroupDestroy(raygen_program_group));
                raygen_program_group = nullptr;
            }
        }
        for (auto& miss_program_group :
             m_program_group_sets.miss_program_groups)
        {
            if (miss_program_group != nullptr)
            {
                optix_check(optixProgramGroupDestroy(miss_program_group));
                miss_program_group = nullptr;
            }
        }
        for (auto& hitgroup_program_group :
             m_program_group_sets.hitgroup_program_groups)
        {
            if (hitgroup_program_group != nullptr)
            {
                optix_check(optixProgramGroupDestroy(hitgroup_program_group));
                hitgroup_program_group = nullptr;
            }
        }

        if (m_module != nullptr)
        {
            optix_check(optixModuleDestroy(m_module));
            m_module = nullptr;
        }
    }

    // must be called before render
    // this cannot be placed in constructor since init_render_strategy is pure
    // virtual
    void init()
    {
        init_render_strategy();
        init_sbt();
    }

    virtual void clear_render() {}

    // TODO: add common GUI elements in this function(template method
    // pattern)
    virtual void run_imgui() {}

    virtual void render(const Camera& camera,
                        const DirectionalLight& directional_light,
                        const SceneDevice& scene,
                        const RenderLayers& render_layers,
                        const OptixTraversableHandle& ias_handle) = 0;

   protected:
    virtual void init_render_strategy() = 0;

    void init_sbt()
    {
        sbt_record_set = optix_create_sbt_records(m_program_group_sets);
        sbt = optix_create_sbt(sbt_record_set);
    }

    // TODO: this should be defined on Camera?
    static CameraParams get_camera_params(const Camera& camera)
    {
        CameraParams camera_params;
        camera_params.transform =
            create_mat3x4_from_glm(camera.get_transform());
        camera_params.fov = camera.get_fov();
        camera_params.F = camera.get_F();
        camera_params.focus = camera.get_focus();
        return camera_params;
    }

    // TODO: this should be defined on SceneDevice?
    static SceneData get_scene_data(const SceneDevice& scene,
                                    const DirectionalLight& directional_light)
    {
        SceneData scene_data;
        scene_data.vertices = reinterpret_cast<float3*>(scene.get_vertices());
        scene_data.indices = reinterpret_cast<uint3*>(scene.get_indices());
        scene_data.normals = reinterpret_cast<float3*>(scene.get_normals());
        scene_data.texcoords = reinterpret_cast<float2*>(scene.get_texcoords());
        scene_data.materials =
            reinterpret_cast<Material*>(scene.get_materials());
        scene_data.n_materials = scene.get_n_materials();
        scene_data.material_ids =
            reinterpret_cast<uint*>(scene.get_material_ids());
        scene_data.textures =
            reinterpret_cast<TextureHeader*>(scene.get_textures());
        scene_data.n_textures = scene.get_n_textures();
        scene_data.n_vertices =
            reinterpret_cast<uint*>(scene.get_n_vertices_buffer());
        scene_data.n_faces =
            reinterpret_cast<uint*>(scene.get_n_faces_buffer());
        scene_data.geometry_ids =
            reinterpret_cast<uint*>(scene.get_geometry_ids());
        scene_data.object_to_worlds =
            reinterpret_cast<Matrix3x4*>(scene.get_object_to_worlds());
        scene_data.world_to_objects =
            reinterpret_cast<Matrix3x4*>(scene.get_world_to_objects());

        scene_data.area_lights =
            reinterpret_cast<AreaLight*>(scene.get_area_lights());
        scene_data.n_area_lights = scene.get_n_area_lights();

        scene_data.envmap.width = scene.get_envmap_resolution().x;
        scene_data.envmap.height = scene.get_envmap_resolution().y;
        scene_data.envmap.data = scene.get_envmap();

        scene_data.arhosek = scene.get_arhosek();

        scene_data.directional_light = directional_light;

        return scene_data;
    }

    OptixDeviceContext context;
    bool debug = false;
    OptixModule m_module = nullptr;
    ProgramGroupSet m_program_group_sets = {};
    OptixPipeline m_pipeline = nullptr;

    SbtRecordSet sbt_record_set;
    OptixShaderBindingTable sbt;
};

}  // namespace fredholm