#pragma once
#include <optix.h>

#include "cuda_util.h"
#include "optix_util.h"

namespace fredholm
{

class Renderer
{
   public:
    Renderer()
    {
#ifdef NDEBUG
        constexpr bool debug = false;
#else
        constexpr bool debug = true;
#endif
        CUcontext cu_context;
        cuda_check(cuCtxGetCurrent(&cu_context));

        context = optix_create_context(cu_context, debug);

        module = optix_create_module(context, "pt.ptx", debug);

        std::vector<ProgramGroupEntry> program_group_entries;
        program_group_entries.push_back(
            {OPTIX_PROGRAM_GROUP_KIND_RAYGEN, "rg", module});

        program_group_entries.push_back(
            {OPTIX_PROGRAM_GROUP_KIND_MISS, "radiance", module});
        program_group_entries.push_back(
            {OPTIX_PROGRAM_GROUP_KIND_MISS, "shadow", module});
        program_group_entries.push_back(
            {OPTIX_PROGRAM_GROUP_KIND_MISS, "light", module});

        program_group_entries.push_back(
            {OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "radiance", module});
        program_group_entries.push_back(
            {OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "shadow", module});
        program_group_entries.push_back(
            {OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "light", module});

        program_group_set =
            optix_create_program_group(context, program_group_entries);

        constexpr uint32_t max_trace_depth = 2;
        constexpr uint32_t max_traversal_depth = 2;
        pipeline =
            optix_create_pipeline(context, program_group_set, max_trace_depth,
                                  max_traversal_depth, debug);

        sbt_record_set = optix_create_sbt_records(program_group_set);

        sbt = optix_create_sbt(sbt_record_set.raygen_records,
                               sbt_record_set.miss_records, 1,
                               sbt_record_set.hitgroup_records, 1);

        // TODO: fill build entries
        std::vector<GASBuildEntry> gas_build_entries;
        gas_build_output = optix_create_gas(context, gas_build_entries);

        // TODO: fill build entries
        std::vector<IASBuildEntry> ias_build_entries;
        ias_build_output = optix_create_ias(context, ias_build_entries);
    }

    ~Renderer()
    {
        cuda_check(cuMemFree(ias_build_output.instance_buffer));
        cuda_check(cuMemFree(ias_build_output.output_buffer));

        for (const auto& gas_output : gas_build_output)
        {
            cuda_check(cuMemFree(gas_output.output_buffer));
        }

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

   private:
    uint32_t width = 0;
    uint32_t height = 0;

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