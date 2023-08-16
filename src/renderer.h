#pragma once
#include <optix.h>

#include "camera.h"
#include "cuda_util.h"
#include "optix_util.h"
#include "render_strategy/render_strategy.h"
#include "scene.h"
#include "shared.h"

namespace fredholm
{

class Renderer
{
   public:
    Renderer(CUcontext cu_context)
    {
#ifdef NDEBUG
        constexpr bool debug = false;
#else
        constexpr bool debug = true;
#endif
        context = optix_create_context(cu_context, debug);
    }

    ~Renderer()
    {
        destroy_ias();
        destroy_gas();

        cuda_check(cuMemFree(sbt_record_set.raygen_records));
        cuda_check(cuMemFree(sbt_record_set.miss_records));
        cuda_check(cuMemFree(sbt_record_set.hitgroup_records));

        optix_check(optixDeviceContextDestroy(context));
    }

    // TODO: maybe optix context could be placed outside renderer?
    OptixDeviceContext get_optix_context() const { return context; }

    // TODO: move these inside SceneDevice?
    void set_scene(const SceneGraph& scene)
    {
        destroy_ias();
        destroy_gas();

        // compile scene graph
        const CompiledScene compiled_scene = scene.compile();

        // build GAS
        std::vector<GASBuildEntry> gas_build_entries;
        for (const auto& geometry : compiled_scene.geometry_nodes)
        {
            GASBuildEntry entry;

            cuda_check(
                cuMemAlloc(&entry.vertex_buffer,
                           geometry->m_vertices.size() * sizeof(float3)));
            cuda_check(
                cuMemcpyHtoD(entry.vertex_buffer, geometry->m_vertices.data(),
                             geometry->m_vertices.size() * sizeof(float3)));
            entry.vertex_count = geometry->m_vertices.size();

            cuda_check(cuMemAlloc(&entry.index_buffer,
                                  geometry->m_indices.size() * sizeof(uint3)));
            cuda_check(
                cuMemcpyHtoD(entry.index_buffer, geometry->m_indices.data(),
                             geometry->m_indices.size() * sizeof(uint3)));
            entry.index_count = geometry->m_indices.size();

            gas_build_entries.push_back(entry);
        }

        gas_build_output = optix_create_gas(context, gas_build_entries);

        // build IAS
        std::vector<IASBuildEntry> ias_build_entries;
        for (int i = 0; i < compiled_scene.geometry_nodes.size(); ++i)
        {
            const auto& geometry = compiled_scene.geometry_nodes[i];
            const auto& transform = compiled_scene.geometry_transforms[i];

            IASBuildEntry entry;
            entry.gas_handle = gas_build_output[i].handle;

            entry.transform[0] = transform[0][0];
            entry.transform[1] = transform[1][0];
            entry.transform[2] = transform[2][0];
            entry.transform[3] = transform[3][0];
            entry.transform[4] = transform[0][1];
            entry.transform[5] = transform[1][1];
            entry.transform[6] = transform[2][1];
            entry.transform[7] = transform[3][1];
            entry.transform[8] = transform[0][2];
            entry.transform[9] = transform[1][2];
            entry.transform[10] = transform[2][2];
            entry.transform[11] = transform[3][2];

            // TODO: set appriopriate value
            entry.sbt_offset = 0;

            ias_build_entries.push_back(entry);
        }

        ias_build_output = optix_create_ias(context, ias_build_entries);

        // create global scene data
        std::vector<float3> vertices;
        std::vector<uint3> indices;
        std::vector<float3> normals;
        std::vector<float2> texcoords;
        std::vector<uint> indices_offset;           // key: geometry id
        std::vector<uint> geometry_ids;             // key: instance id(OptiX)
        std::vector<Matrix3x4> transforms;          // key: instance id(OptiX)
        std::vector<Matrix3x4> inverse_transforms;  // key: instance id(OptiX)

        for (int i = 0; i < compiled_scene.geometry_nodes.size(); ++i)
        {
            const auto& geometry = compiled_scene.geometry_nodes[i];

            indices_offset.push_back(indices.size());
            geometry_ids.push_back(i);

            vertices.insert(vertices.end(), geometry->m_vertices.begin(),
                            geometry->m_vertices.end());
            indices.insert(indices.end(), geometry->m_indices.begin(),
                           geometry->m_indices.end());
            normals.insert(normals.end(), geometry->m_normals.begin(),
                           geometry->m_normals.end());
            texcoords.insert(texcoords.end(), geometry->m_texcoords.begin(),
                             geometry->m_texcoords.end());

            transforms.push_back(
                create_mat3x4_from_glm(compiled_scene.geometry_transforms[i]));
            inverse_transforms.push_back(create_mat3x4_from_glm(
                glm::inverse(compiled_scene.geometry_transforms[i])));
        }

        for (int i = 0; i < compiled_scene.instance_nodes.size(); ++i)
        {
            const auto& instance = compiled_scene.instance_nodes[i];

            // find geometry id
            for (int j = 0; j < compiled_scene.geometry_nodes.size(); ++j)
            {
                if (compiled_scene.geometry_nodes[j] == instance->geometry)
                {
                    geometry_ids.push_back(j);
                    break;
                }
            }

            transforms.push_back(
                create_mat3x4_from_glm(compiled_scene.instance_transforms[i]));
            inverse_transforms.push_back(create_mat3x4_from_glm(
                glm::inverse(compiled_scene.instance_transforms[i])));
        }

        // allocate scene data on device
        destroy_scene_data();

        cuda_check(
            cuMemAlloc(&vertices_buffer, vertices.size() * sizeof(float3)));
        cuda_check(cuMemcpyHtoD(vertices_buffer, vertices.data(),
                                vertices.size() * sizeof(float3)));

        cuda_check(cuMemAlloc(&indices_buffer, indices.size() * sizeof(uint3)));
        cuda_check(cuMemcpyHtoD(indices_buffer, indices.data(),
                                indices.size() * sizeof(uint3)));

        cuda_check(
            cuMemAlloc(&normals_buffer, normals.size() * sizeof(float3)));
        cuda_check(cuMemcpyHtoD(normals_buffer, normals.data(),
                                normals.size() * sizeof(float3)));

        cuda_check(
            cuMemAlloc(&texcoords_buffer, texcoords.size() * sizeof(float2)));
        cuda_check(cuMemcpyHtoD(texcoords_buffer, texcoords.data(),
                                texcoords.size() * sizeof(float2)));

        cuda_check(cuMemAlloc(&indices_offset_buffer,
                              indices_offset.size() * sizeof(uint)));
        cuda_check(cuMemcpyHtoD(indices_offset_buffer, indices_offset.data(),
                                indices_offset.size() * sizeof(uint)));

        cuda_check(cuMemAlloc(&geometry_ids_buffer,
                              geometry_ids.size() * sizeof(uint)));
        cuda_check(cuMemcpyHtoD(geometry_ids_buffer, geometry_ids.data(),
                                geometry_ids.size() * sizeof(uint)));

        cuda_check(cuMemAlloc(&object_to_world_buffer,
                              transforms.size() * sizeof(Matrix3x4)));
        cuda_check(cuMemcpyHtoD(object_to_world_buffer, transforms.data(),
                                transforms.size() * sizeof(Matrix3x4)));

        cuda_check(cuMemAlloc(&world_to_object_buffer,
                              inverse_transforms.size() * sizeof(Matrix3x4)));
        cuda_check(cuMemcpyHtoD(world_to_object_buffer,
                                inverse_transforms.data(),
                                inverse_transforms.size() * sizeof(Matrix3x4)));
    }

    void set_render_strategy(RenderStrategy* strategy)
    {
        m_render_strategy = strategy;

        sbt_record_set = optix_create_sbt_records(
            m_render_strategy->get_program_group_sets());

        sbt = optix_create_sbt(sbt_record_set);
    }

    void render(uint32_t width, uint32_t height, const Camera& camera,
                const CUdeviceptr& beauty)
    {
        if (m_render_strategy)
        {
            CameraParams camera_params;
            camera_params.transform =
                create_mat3x4_from_glm(camera.m_transform);
            camera_params.fov = camera.m_fov;
            camera_params.F = camera.m_F;
            camera_params.focus = camera.m_focus;

            SceneData scene_data;
            scene_data.vertices = reinterpret_cast<float3*>(vertices_buffer);
            scene_data.indices = reinterpret_cast<uint3*>(indices_buffer);
            scene_data.normals = reinterpret_cast<float3*>(normals_buffer);
            scene_data.texcoords = reinterpret_cast<float2*>(texcoords_buffer);
            scene_data.indices_offsets =
                reinterpret_cast<uint*>(indices_offset_buffer);
            scene_data.geometry_ids =
                reinterpret_cast<uint*>(geometry_ids_buffer);
            scene_data.object_to_worlds =
                reinterpret_cast<Matrix3x4*>(object_to_world_buffer);
            scene_data.world_to_objects =
                reinterpret_cast<Matrix3x4*>(world_to_object_buffer);

            m_render_strategy->render(width, height, camera_params, scene_data,
                                      ias_build_output.handle, sbt, beauty);
        }
    }

    void synchronize() const { cuda_check(cuCtxSynchronize()); }

   private:
    void destroy_gas()
    {
        for (auto& gas_output : gas_build_output)
        {
            if (gas_output.output_buffer != 0)
            {
                cuda_check(cuMemFree(gas_output.output_buffer));
                gas_output.output_buffer = 0;
            }
            gas_output.handle = 0;
        }
    }

    void destroy_ias()
    {
        if (ias_build_output.instance_buffer != 0)
        {
            cuda_check(cuMemFree(ias_build_output.instance_buffer));
            ias_build_output.instance_buffer = 0;
        }
        if (ias_build_output.output_buffer != 0)
        {
            cuda_check(cuMemFree(ias_build_output.output_buffer));
            ias_build_output.output_buffer = 0;
        }
        ias_build_output.handle = 0;
    }

    void destroy_scene_data()
    {
        if (vertices_buffer != 0)
        {
            cuda_check(cuMemFree(vertices_buffer));
            vertices_buffer = 0;
        }
        if (indices_buffer != 0)
        {
            cuda_check(cuMemFree(indices_buffer));
            indices_buffer = 0;
        }
        if (normals_buffer != 0)
        {
            cuda_check(cuMemFree(normals_buffer));
            normals_buffer = 0;
        }
        if (texcoords_buffer != 0)
        {
            cuda_check(cuMemFree(texcoords_buffer));
            texcoords_buffer = 0;
        }
        if (indices_offset_buffer != 0)
        {
            cuda_check(cuMemFree(indices_offset_buffer));
            indices_offset_buffer = 0;
        }
        if (geometry_ids_buffer != 0)
        {
            cuda_check(cuMemFree(geometry_ids_buffer));
            geometry_ids_buffer = 0;
        }
        if (object_to_world_buffer != 0)
        {
            cuda_check(cuMemFree(object_to_world_buffer));
            object_to_world_buffer = 0;
        }
        if (world_to_object_buffer != 0)
        {
            cuda_check(cuMemFree(world_to_object_buffer));
            world_to_object_buffer = 0;
        }
    }

    static Matrix3x4 create_mat3x4_from_glm(const glm::mat4& m)
    {
        return make_mat3x4(make_float4(m[0][0], m[1][0], m[2][0], m[3][0]),
                           make_float4(m[0][1], m[1][1], m[2][1], m[3][1]),
                           make_float4(m[0][2], m[1][2], m[2][2], m[3][2]));
    }

    OptixDeviceContext context;

    OptixModule module = nullptr;
    ProgramGroupSet program_group_set;
    OptixPipeline pipeline = nullptr;
    SbtRecordSet sbt_record_set;
    OptixShaderBindingTable sbt;
    std::vector<GASBuildOutput> gas_build_output;
    IASBuildOutput ias_build_output;

    // TODO: move these inside SceneDevice?
    // global scene data
    CUdeviceptr vertices_buffer = 0;
    CUdeviceptr indices_buffer = 0;
    CUdeviceptr normals_buffer = 0;
    CUdeviceptr texcoords_buffer = 0;
    CUdeviceptr indices_offset_buffer = 0;
    CUdeviceptr geometry_ids_buffer = 0;
    CUdeviceptr object_to_world_buffer = 0;
    CUdeviceptr world_to_object_buffer = 0;

    RenderStrategy* m_render_strategy = nullptr;
};

}  // namespace fredholm