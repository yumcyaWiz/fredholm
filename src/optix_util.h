#pragma once
#ifndef __CUDACC__
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <source_location>
#include <vector>

#include "cuda_util.h"
#endif

namespace fredholm
{

template <typename T>
struct SbtRecord
{
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayGenSbtRecordData
{
};

struct MissSbtRecordData
{
};

struct HitGroupSbtRecordData
{
};

using RayGenSbtRecord = SbtRecord<RayGenSbtRecordData>;
using MissSbtRecord = SbtRecord<MissSbtRecordData>;
using HitGroupSbtRecord = SbtRecord<HitGroupSbtRecordData>;

#ifndef __CUDACC__
struct SbtRecordSet
{
    CUdeviceptr raygen_records = 0;
    uint32_t raygen_records_count = 1;

    CUdeviceptr miss_records = 0;
    uint32_t miss_records_count = 1;

    CUdeviceptr hitgroup_records = 0;
    uint32_t hitgroup_records_count = 1;
};

inline std::vector<char> read_binary_from_file(
    const std::filesystem::path& filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error(
            std::format("Failed to open file: {}", filepath.string()));
    }

    std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});
    if (buffer.size() == 0)
    {
        throw std::runtime_error(
            std::format("Failed to read file: {}", filepath.string()));
    }

    return buffer;
}

inline void optix_check_log(
    const OptixResult& result, const char* log, size_t sizeof_log,
    const std::source_location& loc = std::source_location::current())
{
    if (result == OPTIX_SUCCESS) return;
    throw std::runtime_error(
        std::format("OptiX error: {}({}:{}): {}\n", loc.file_name(), loc.line(),
                    loc.column(), std::string(log, sizeof_log)));
}

inline void optix_check(
    const OptixResult& result,
    const std::source_location& loc = std::source_location::current())
{
    if (result == OPTIX_SUCCESS) return;
    throw std::runtime_error(
        std::format("OptiX error: {}({}:{}): {}\n", loc.file_name(), loc.line(),
                    loc.column(), optixGetErrorString(result)));
}

inline void optix_log_callback(unsigned int level, const char* tag,
                               const char* message, void* cbdata)
{
    if (level == 1)
    {
        std::cerr << std::format("Optix critical error[{}]: {}\n", tag, message)
                  << std::endl;
    }
    else if (level == 2)
    {
        std::cerr << std::format("Optix error[{}]: {}\n", tag, message)
                  << std::endl;
    }
    else if (level == 3)
    {
        std::cerr << std::format("Optix warning[{}]: {}\n", tag, message)
                  << std::endl;
    }
    else if (level == 4)
    {
        std::cerr << std::format("Optix info[{}]: {}\n", tag, message)
                  << std::endl;
    }
}

inline OptixDeviceContext optix_create_context(const CUcontext& cu_context,
                                               bool debug = false)
{
    OptixDeviceContextOptions options = {};
    options.validationMode = debug ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
                                   : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    options.logCallbackFunction = &optix_log_callback;
    options.logCallbackLevel = 4;

    OptixDeviceContext context = nullptr;
    optix_check(optixDeviceContextCreate(cu_context, &options, &context));

    return context;
}

inline OptixPipelineCompileOptions optix_create_pipeline_compile_options(
    bool debug = false)
{
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues = 3;
    pipeline_compile_options.numAttributeValues = 3;
    pipeline_compile_options.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.exceptionFlags =
        debug ? (OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                 OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_USER)
              : OPTIX_EXCEPTION_FLAG_NONE;
    return pipeline_compile_options;
}

inline OptixModule optix_create_module(const OptixDeviceContext& context,
                                       const std::filesystem::path& filepath,
                                       bool debug = false)
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.debugLevel =
        debug ? OPTIX_COMPILE_DEBUG_LEVEL_FULL : OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    module_compile_options.optLevel = debug
                                          ? OPTIX_COMPILE_OPTIMIZATION_LEVEL_0
                                          : OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

    const std::vector<char> ptx = read_binary_from_file(filepath);

    OptixPipelineCompileOptions pipeline_compile_options =
        optix_create_pipeline_compile_options(debug);

    OptixModule module = nullptr;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    optix_check_log(
        optixModuleCreate(context, &module_compile_options,
                          &pipeline_compile_options, ptx.data(),
                          ptx.size() * sizeof(char), log, &sizeof_log, &module),
        log, sizeof_log);

    return module;
}

struct ProgramGroupEntry
{
    OptixProgramGroupKind kind;
    std::string entry_function_name;
    OptixModule module;
};

struct ProgramGroupSet
{
    std::vector<OptixProgramGroup> raygen_program_groups;
    std::vector<OptixProgramGroup> miss_program_groups;
    std::vector<OptixProgramGroup> hitgroup_program_groups;
};

inline ProgramGroupSet optix_create_program_group(
    const OptixDeviceContext& context,
    const std::vector<ProgramGroupEntry>& program_groups)
{
    ProgramGroupSet ret;

    OptixProgramGroupOptions program_group_options = {};

    for (const ProgramGroupEntry& prog_entry : program_groups)
    {
        OptixProgramGroupDesc prog_group_desc = {};

        const std::string raygen_entry_function_name =
            "__raygen__" + prog_entry.entry_function_name;
        const std::string miss_entry_function_name =
            "__miss__" + prog_entry.entry_function_name;
        const std::string anyhit_entry_function_name =
            "__anyhit__" + prog_entry.entry_function_name;
        const std::string closest_entry_function_name =
            "__closesthit__" + prog_entry.entry_function_name;

        prog_group_desc.kind = prog_entry.kind;
        if (prog_entry.kind == OPTIX_PROGRAM_GROUP_KIND_RAYGEN)
        {
            prog_group_desc.raygen.module = prog_entry.module;
            prog_group_desc.raygen.entryFunctionName =
                raygen_entry_function_name.c_str();
        }
        else if (prog_entry.kind == OPTIX_PROGRAM_GROUP_KIND_MISS)
        {
            prog_group_desc.miss.module = prog_entry.module;
            prog_group_desc.miss.entryFunctionName =
                miss_entry_function_name.c_str();
        }
        else if (prog_entry.kind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP)
        {
            prog_group_desc.hitgroup.moduleAH = prog_entry.module;
            prog_group_desc.hitgroup.entryFunctionNameAH =
                anyhit_entry_function_name.c_str();
            prog_group_desc.hitgroup.moduleCH = prog_entry.module;
            prog_group_desc.hitgroup.entryFunctionNameCH =
                closest_entry_function_name.c_str();
        }
        else { throw std::runtime_error("Invalid program group kind"); }

        OptixProgramGroup program_group = nullptr;
        char log[2048];
        size_t sizeof_log = sizeof(log);
        optix_check_log(optixProgramGroupCreate(context, &prog_group_desc, 1,
                                                &program_group_options, log,
                                                &sizeof_log, &program_group),
                        log, sizeof_log);

        if (prog_entry.kind == OPTIX_PROGRAM_GROUP_KIND_RAYGEN)
        {
            ret.raygen_program_groups.push_back(program_group);
        }
        else if (prog_entry.kind == OPTIX_PROGRAM_GROUP_KIND_MISS)
        {
            ret.miss_program_groups.push_back(program_group);
        }
        else if (prog_entry.kind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP)
        {
            ret.hitgroup_program_groups.push_back(program_group);
        }
    }

    return ret;
}

inline OptixPipeline optix_create_pipeline(
    const OptixDeviceContext& context,
    const ProgramGroupSet& program_group_output, uint32_t max_trace_depth = 1,
    uint32_t max_traversal_depth = 1, bool debug = false)
{
    OptixPipelineCompileOptions pipeline_compile_options =
        optix_create_pipeline_compile_options(debug);

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;

    std::vector<OptixProgramGroup> program_groups;
    for (const auto& raygen_program_group :
         program_group_output.raygen_program_groups)
    {
        program_groups.push_back(raygen_program_group);
    }
    for (const auto& miss_program_group :
         program_group_output.miss_program_groups)
    {
        program_groups.push_back(miss_program_group);
    }
    for (const auto& hitgroup_program_group :
         program_group_output.hitgroup_program_groups)
    {
        program_groups.push_back(hitgroup_program_group);
    }

    OptixPipeline pipeline = nullptr;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    optix_check_log(
        optixPipelineCreate(context, &pipeline_compile_options,
                            &pipeline_link_options, program_groups.data(),
                            program_groups.size(), log, &sizeof_log, &pipeline),
        log, sizeof_log);

    OptixStackSizes stack_sizes = {};
    for (auto& program_group : program_groups)
    {
        optix_check_log(optixUtilAccumulateStackSizes(program_group,
                                                      &stack_sizes, pipeline),
                        log, sizeof_log);
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    optix_check_log(
        optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                   0,  // maxCCDepth
                                   0,  // maxDCDEpth
                                   &direct_callable_stack_size_from_traversal,
                                   &direct_callable_stack_size_from_state,
                                   &continuation_stack_size),
        log, sizeof_log);

    optix_check_log(optixPipelineSetStackSize(
                        pipeline, direct_callable_stack_size_from_traversal,
                        direct_callable_stack_size_from_state,
                        continuation_stack_size, max_traversal_depth),
                    log, sizeof_log);

    return pipeline;
}

inline SbtRecordSet optix_create_sbt_records(
    const ProgramGroupSet& program_groups)
{
    std::vector<RayGenSbtRecord> raygen_records;
    std::vector<MissSbtRecord> miss_records;
    std::vector<HitGroupSbtRecord> hit_records;
    for (const OptixProgramGroup& program_group :
         program_groups.raygen_program_groups)
    {
        RayGenSbtRecord raygen_record;
        optix_check(optixSbtRecordPackHeader(program_group, &raygen_record));
        raygen_records.push_back(raygen_record);
    }
    for (const OptixProgramGroup& program_group :
         program_groups.miss_program_groups)
    {
        MissSbtRecord miss_record;
        optix_check(optixSbtRecordPackHeader(program_group, &miss_record));
        miss_records.push_back(miss_record);
    }
    for (const OptixProgramGroup& program_group :
         program_groups.hitgroup_program_groups)
    {
        HitGroupSbtRecord hit_record;
        optix_check(optixSbtRecordPackHeader(program_group, &hit_record));
        hit_records.push_back(hit_record);
    }

    SbtRecordSet ret;
    ret.raygen_records_count = 1;
    if (ret.raygen_records_count > 0)
    {
        cuda_check(cuMemAlloc(&ret.raygen_records,
                              sizeof(RayGenSbtRecord) * raygen_records.size()));
        cuda_check(
            cuMemcpyHtoD(ret.raygen_records, raygen_records.data(),
                         sizeof(RayGenSbtRecord) * raygen_records.size()));
    }

    ret.miss_records_count = miss_records.size();
    if (ret.miss_records_count > 0)
    {
        cuda_check(cuMemAlloc(&ret.miss_records,
                              sizeof(MissSbtRecord) * miss_records.size()));
        cuda_check(cuMemcpyHtoD(ret.miss_records, miss_records.data(),
                                sizeof(MissSbtRecord) * miss_records.size()));
    }

    ret.hitgroup_records_count = hit_records.size();
    if (ret.hitgroup_records_count > 0)
    {
        cuda_check(cuMemAlloc(&ret.hitgroup_records,
                              sizeof(HitGroupSbtRecord) * hit_records.size()));
        cuda_check(
            cuMemcpyHtoD(ret.hitgroup_records, hit_records.data(),
                         sizeof(HitGroupSbtRecord) * hit_records.size()));
    }

    return ret;
}

inline OptixShaderBindingTable optix_create_sbt(
    const SbtRecordSet& sbt_record_set)
{
    OptixShaderBindingTable ret;
    ret.raygenRecord = sbt_record_set.raygen_records;

    ret.missRecordBase = sbt_record_set.miss_records;
    ret.missRecordStrideInBytes = sizeof(MissSbtRecord);
    ret.missRecordCount = sbt_record_set.miss_records_count;

    ret.hitgroupRecordBase = sbt_record_set.hitgroup_records;
    ret.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    ret.hitgroupRecordCount = sbt_record_set.hitgroup_records_count;

    ret.callablesRecordBase = 0;
    ret.callablesRecordStrideInBytes = 0;
    ret.callablesRecordCount = 0;

    return ret;
}

struct GASBuildEntry
{
    CUdeviceptr vertex_buffer = 0;
    uint32_t vertex_count = 0;
    CUdeviceptr index_buffer = 0;
    uint32_t index_count = 0;
};

struct GASBuildOutput
{
    CUdeviceptr output_buffer = 0;
    OptixTraversableHandle handle = 0;
};

inline std::vector<GASBuildOutput> optix_create_gas(
    const OptixDeviceContext& context,
    const std::vector<GASBuildEntry>& build_entries)
{
    std::vector<GASBuildOutput> ret;

    OptixAccelBuildOptions accel_build_options = {};
    accel_build_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_build_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

    for (const auto& build_entry : build_entries)
    {
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
        build_input.triangleArray.numVertices = build_entry.vertex_count;
        build_input.triangleArray.vertexBuffers = &build_entry.vertex_buffer;

        build_input.triangleArray.indexFormat =
            OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
        build_input.triangleArray.numIndexTriplets =
            build_entry.index_count / 3;
        build_input.triangleArray.indexBuffer = build_entry.index_buffer;

        build_input.triangleArray.flags = flags;
        build_input.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        optix_check(optixAccelComputeMemoryUsage(
            context, &accel_build_options, &build_input, 1, &gas_buffer_sizes));

        CUdeviceptr temp_buffer = 0;
        cuda_check(cuMemAlloc(&temp_buffer, gas_buffer_sizes.tempSizeInBytes));
        CUdeviceptr output_buffer = 0;
        cuda_check(
            cuMemAlloc(&output_buffer, gas_buffer_sizes.outputSizeInBytes));

        OptixTraversableHandle gas_handle;
        optix_check(optixAccelBuild(
            context, 0, &accel_build_options, &build_input, 1, temp_buffer,
            gas_buffer_sizes.tempSizeInBytes, output_buffer,
            gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));

        ret.push_back(GASBuildOutput{output_buffer, gas_handle});
    }

    return ret;
}

struct IASBuildEntry
{
    OptixTraversableHandle gas_handle;
    float transform[12];
    uint32_t sbt_offset;
};

struct IASBuildOutput
{
    CUdeviceptr instance_buffer = 0;
    CUdeviceptr output_buffer = 0;
    OptixTraversableHandle handle = 0;
};

inline IASBuildOutput optix_create_ias(
    const OptixDeviceContext& context,
    const std::vector<IASBuildEntry>& build_entries)
{
    OptixAccelBuildOptions accel_build_options = {};
    accel_build_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_build_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    std::vector<OptixInstance> instances;
    for (int i = 0; i < build_entries.size(); ++i)
    {
        OptixInstance instance = {};
        memcpy(instance.transform, build_entries[i].transform,
               sizeof(float) * 12);
        instance.instanceId = i;
        instance.visibilityMask = 1;
        instance.sbtOffset = build_entries[i].sbt_offset;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = build_entries[i].gas_handle;
        instances.push_back(instance);
    }

    CUdeviceptr instance_buffer = 0;
    if (instances.size() > 0)
    {
        cuda_check(cuMemAlloc(&instance_buffer,
                              sizeof(OptixInstance) * instances.size()));
        cuda_check(cuMemcpyHtoD(instance_buffer, instances.data(),
                                sizeof(OptixInstance) * instances.size()));
    }

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = instance_buffer;
    build_input.instanceArray.numInstances = instances.size();

    OptixAccelBufferSizes ias_buffer_sizes;
    optix_check(optixAccelComputeMemoryUsage(
        context, &accel_build_options, &build_input, 1, &ias_buffer_sizes));

    CUdeviceptr temp_buffer = 0;
    cuda_check(cuMemAlloc(&temp_buffer, ias_buffer_sizes.tempSizeInBytes));
    CUdeviceptr output_buffer = 0;
    cuda_check(cuMemAlloc(&output_buffer, ias_buffer_sizes.outputSizeInBytes));
    OptixTraversableHandle ias_handle;
    optix_check(optixAccelBuild(
        context, 0, &accel_build_options, &build_input, 1, temp_buffer,
        ias_buffer_sizes.tempSizeInBytes, output_buffer,
        ias_buffer_sizes.outputSizeInBytes, &ias_handle, nullptr, 0));

    return IASBuildOutput{instance_buffer, output_buffer, ias_handle};
}

#endif

}  // namespace fredholm