#pragma once
#include <optix.h>
#include <optix_stack_size.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <source_location>
#include <vector>

#include "cuda_util.h"

namespace fredholm
{

inline std::vector<uint8_t> read_binary_from_file(
    const std::filesystem::path& filepath)
{
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(
            std::format("Failed to open file: {}", filepath.string()));
    }

    std::vector<uint8_t> buffer(std::istreambuf_iterator<char>(file), {});
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

    const std::vector<uint8_t> ptx = read_binary_from_file(filepath);

    OptixPipelineCompileOptions pipeline_compile_options =
        optix_create_pipeline_compile_options(debug);
    OptixModule module = nullptr;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    optix_check_log(optixModuleCreate(context, &module_compile_options,
                                      &pipeline_compile_options,
                                      reinterpret_cast<const char*>(ptx.data()),
                                      ptx.size(), log, &sizeof_log, &module),
                    log, sizeof_log);

    return module;
}

enum class ProgramGroupKind { RAYGEN, MISS, HITGROUP };

struct ProgramGroupEntry {
    ProgramGroupKind kind;
    std::string entry_function_name;
    OptixModule module;
};

inline std::vector<OptixProgramGroup> optix_create_program_group(
    const OptixDeviceContext& context, const OptixModule& module,
    const std::vector<ProgramGroupEntry>& program_groups)
{
    OptixProgramGroupOptions program_group_options = {};

    std::vector<OptixProgramGroup> ret;
    for (const ProgramGroupEntry& prog_entry : program_groups) {
        OptixProgramGroupDesc prog_group_desc = {};

        const std::string raygen_entry_function_name =
            "__raygen__" + prog_entry.entry_function_name;
        const std::string miss_entry_function_name =
            "__miss__" + prog_entry.entry_function_name;
        const std::string anyhit_entry_function_name =
            "__anyhit__" + prog_entry.entry_function_name;
        const std::string closest_entry_function_name =
            "__closesthit__" + prog_entry.entry_function_name;

        if (prog_entry.kind == ProgramGroupKind::RAYGEN) {
            prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            prog_group_desc.raygen.module = module;
            prog_group_desc.raygen.entryFunctionName =
                raygen_entry_function_name.c_str();
        } else if (prog_entry.kind == ProgramGroupKind::MISS) {
            prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            prog_group_desc.miss.module = module;
            prog_group_desc.miss.entryFunctionName =
                miss_entry_function_name.c_str();
        } else if (prog_entry.kind == ProgramGroupKind::HITGROUP) {
            prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            prog_group_desc.hitgroup.moduleAH = module;
            prog_group_desc.hitgroup.entryFunctionNameAH =
                anyhit_entry_function_name.c_str();
            prog_group_desc.hitgroup.moduleCH = module;
            prog_group_desc.hitgroup.entryFunctionNameCH =
                closest_entry_function_name.c_str();
        } else {
            throw std::runtime_error("Invalid program group kind");
        }

        OptixProgramGroup program_group = nullptr;
        char log[2048];
        size_t sizeof_log = sizeof(log);
        optix_check_log(optixProgramGroupCreate(context, &prog_group_desc, 1,
                                                &program_group_options, log,
                                                &sizeof_log, &program_group),
                        log, sizeof_log);
        ret.push_back(program_group);
    }

    return ret;
}

inline OptixPipeline optix_create_pipeline(
    const OptixDeviceContext& context,
    const std::vector<OptixProgramGroup>& program_groups,
    uint32_t max_trace_depth = 1, uint32_t max_traversal_depth = 1,
    bool debug = false)
{
    OptixPipelineCompileOptions pipeline_compile_options =
        optix_create_pipeline_compile_options(debug);

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;

    OptixPipeline pipeline = nullptr;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    optix_check_log(
        optixPipelineCreate(context, &pipeline_compile_options,
                            &pipeline_link_options, program_groups.data(),
                            program_groups.size(), log, &sizeof_log, &pipeline),
        log, sizeof_log);

    OptixStackSizes stack_sizes = {};
    for (auto& program_group : program_groups) {
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

struct RayGenSbtRecordData {
};

struct MissSbtRecordData {
};

struct HitGroupSbtRecordData {
    uint3* indices;
    uint* material_ids;
};

template <typename T>
struct SbtRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenSbtRecord = SbtRecord<RayGenSbtRecordData>;
using MissSbtRecord = SbtRecord<MissSbtRecordData>;
using HitGroupSbtRecord = SbtRecord<HitGroupSbtRecordData>;

inline void optix_create_shader_binding_table() {}

}  // namespace fredholm