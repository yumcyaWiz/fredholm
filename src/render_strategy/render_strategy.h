#pragma once
#include "camera.h"
#include "optix_util.h"
#include "scene.h"
#include "shared.h"
#include "util.h"

namespace fredholm
{

struct RenderLayers
{
    // float4 buffer
    CUdeviceptr beauty = 0;
    CUdeviceptr position = 0;
    CUdeviceptr normal = 0;
    CUdeviceptr texcoord = 0;
    CUdeviceptr albedo = 0;
    CUdeviceptr material_id = 0;
};

// TODO: add parameters specific to each strategy

// TODO: change AOV based on strategy
class RenderStrategy
{
   public:
    RenderStrategy() = default;

    virtual ~RenderStrategy()
    {
        if (m_params_buffer != 0)
        {
            cuda_check(cuMemFree(m_params_buffer));
            m_params_buffer = 0;
        }

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

    const ProgramGroupSet& get_program_group_sets() const
    {
        return m_program_group_sets;
    }

    virtual void runImGui() = 0;

    virtual void render(uint32_t width, uint32_t height, const Camera& camera,
                        const SceneDevice& scene,
                        const OptixTraversableHandle& ias_handle,
                        const OptixShaderBindingTable& sbt,
                        const RenderLayers& layers) = 0;

   protected:
    // this should be defined on Camera
    static CameraParams get_camera_params(const Camera& camera)
    {
        CameraParams camera_params;
        camera_params.transform = create_mat3x4_from_glm(camera.m_transform);
        camera_params.fov = camera.m_fov;
        camera_params.F = camera.m_F;
        camera_params.focus = camera.m_focus;
        return camera_params;
    }

    // TODO: this should be defined on SceneDevice
    static SceneData get_scene_data(const SceneDevice& scene)
    {
        SceneData scene_data;
        scene_data.vertices = reinterpret_cast<float3*>(scene.get_vertices());
        scene_data.indices = reinterpret_cast<uint3*>(scene.get_indices());
        scene_data.normals = reinterpret_cast<float3*>(scene.get_normals());
        scene_data.texcoords = reinterpret_cast<float2*>(scene.get_texcoords());
        scene_data.materials =
            reinterpret_cast<Material*>(scene.get_materials());
        scene_data.material_ids =
            reinterpret_cast<uint*>(scene.get_material_ids());
        scene_data.textures =
            reinterpret_cast<TextureHeader*>(scene.get_textures());
        scene_data.indices_offsets =
            reinterpret_cast<uint*>(scene.get_indices_offset());
        scene_data.geometry_ids =
            reinterpret_cast<uint*>(scene.get_geometry_ids());
        scene_data.object_to_worlds =
            reinterpret_cast<Matrix3x4*>(scene.get_object_to_worlds());
        scene_data.world_to_objects =
            reinterpret_cast<Matrix3x4*>(scene.get_world_to_objects());
        return scene_data;
    }

    OptixModule m_module = nullptr;
    ProgramGroupSet m_program_group_sets = {};
    OptixPipeline m_pipeline = nullptr;

    CUdeviceptr m_params_buffer = 0;
};

}  // namespace fredholm