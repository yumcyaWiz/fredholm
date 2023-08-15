#pragma once
#include <optix.h>

#include "cuda_util.h"
#include "optix_util.h"
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

        module = optix_create_module(context, "test.ptx", debug);

        std::vector<ProgramGroupEntry> program_group_entries;
        program_group_entries.push_back(
            {OPTIX_PROGRAM_GROUP_KIND_RAYGEN, "rg", module});
        program_group_entries.push_back(
            {OPTIX_PROGRAM_GROUP_KIND_MISS, "", module});

        // program_group_entries.push_back(
        //     {OPTIX_PROGRAM_GROUP_KIND_MISS, "radiance", module});
        // program_group_entries.push_back(
        //     {OPTIX_PROGRAM_GROUP_KIND_MISS, "shadow", module});
        // program_group_entries.push_back(
        //     {OPTIX_PROGRAM_GROUP_KIND_MISS, "light", module});

        // program_group_entries.push_back(
        //     {OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "radiance", module});
        // program_group_entries.push_back(
        //     {OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "shadow", module});
        // program_group_entries.push_back(
        //     {OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "light", module});

        program_group_set =
            optix_create_program_group(context, program_group_entries);

        constexpr uint32_t max_trace_depth = 2;
        constexpr uint32_t max_traversal_depth = 2;
        pipeline =
            optix_create_pipeline(context, program_group_set, max_trace_depth,
                                  max_traversal_depth, debug);

        sbt_record_set = optix_create_sbt_records(program_group_set);

        sbt = optix_create_sbt(sbt_record_set);

        // TODO: fill build entries
        std::vector<GASBuildEntry> gas_build_entries;
        gas_build_output = optix_create_gas(context, gas_build_entries);

        // TODO: fill build entries
        std::vector<IASBuildEntry> ias_build_entries;
        ias_build_output = optix_create_ias(context, ias_build_entries);
    }

    ~Renderer()
    {
        destroy_ias();
        destroy_gas();

        cuda_check(cuMemFree(sbt_record_set.raygen_records));
        cuda_check(cuMemFree(sbt_record_set.miss_records));
        cuda_check(cuMemFree(sbt_record_set.hitgroup_records));

        for (const auto& raygen_program_group :
             program_group_set.raygen_program_groups)
        {
            optix_check(optixProgramGroupDestroy(raygen_program_group));
        }
        for (const auto& miss_program_group :
             program_group_set.miss_program_groups)
        {
            optix_check(optixProgramGroupDestroy(miss_program_group));
        }
        for (const auto& hitgroup_program_group :
             program_group_set.hitgroup_program_groups)
        {
            optix_check(optixProgramGroupDestroy(hitgroup_program_group));
        }

        optix_check(optixPipelineDestroy(pipeline));

        optix_check(optixModuleDestroy(module));

        optix_check(optixDeviceContextDestroy(context));
    }

    void set_scene(const SceneGraph& scene)
    {
        destroy_ias();
        destroy_gas();

        const CompiledScene compiled_scene = scene.compile();

        // build GAS
        std::vector<GASBuildEntry> gas_build_entries;
        for (const auto& geometry : compiled_scene.geometry_nodes)
        {
            GASBuildEntry entry;

            printf("%d\n", geometry->m_vertices.size());
            printf("%d\n", geometry->m_indices.size());

            cuda_check(
                cuMemAlloc(&entry.vertex_buffer,
                           geometry->m_vertices.size() * sizeof(float3)));
            cuda_check(
                cuMemcpyHtoD(entry.vertex_buffer, geometry->m_vertices.data(),
                             geometry->m_vertices.size() * sizeof(float3)));

            cuda_check(cuMemAlloc(&entry.index_buffer,
                                  geometry->m_indices.size() * sizeof(uint3)));
            cuda_check(
                cuMemcpyHtoD(entry.index_buffer, geometry->m_indices.data(),
                             geometry->m_indices.size() * sizeof(uint3)));

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
            entry.transform[3] = transform[0][1];
            entry.transform[4] = transform[1][1];
            entry.transform[5] = transform[2][1];
            entry.transform[6] = transform[0][2];
            entry.transform[7] = transform[1][2];
            entry.transform[8] = transform[2][2];
            entry.transform[9] = transform[0][3];
            entry.transform[10] = transform[1][3];
            entry.transform[11] = transform[2][3];
            // TODO: set appriopriate value
            entry.sbt_offset = 1;

            ias_build_entries.push_back(entry);
        }

        ias_build_output = optix_create_ias(context, ias_build_entries);
    }

    void render(uint32_t width, uint32_t height, const CUdeviceptr& beauty)
    {
        LaunchParams params;
        params.width = width;
        params.height = height;
        params.render_layer.beauty = reinterpret_cast<float4*>(beauty);

        CUdeviceptr params_buffer;
        cuda_check(cuMemAlloc(&params_buffer, sizeof(LaunchParams)));
        cuda_check(cuMemcpyHtoD(params_buffer, &params, sizeof(LaunchParams)));

        optix_check(optixLaunch(pipeline, 0, params_buffer,
                                sizeof(LaunchParams), &sbt, width, height, 1));
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
    }

    OptixDeviceContext context;

    OptixModule module = nullptr;
    ProgramGroupSet program_group_set;
    OptixPipeline pipeline = nullptr;
    SbtRecordSet sbt_record_set;
    OptixShaderBindingTable sbt;
    std::vector<GASBuildOutput> gas_build_output;
    IASBuildOutput ias_build_output;
};

}  // namespace fredholm