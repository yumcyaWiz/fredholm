#pragma once
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include "device/buffer.h"
#include "device/util.h"
#include "scene.h"

template <typename RayGenSbtRecord, typename MissSbtRecord,
          typename HitGroupSbtRecord>
class Renderer
{
 public:
  Renderer(uint32_t width, uint32_t height, bool enable_validation_mode = false)
      : m_width(width),
        m_height(height),
        m_enable_validation_mode(enable_validation_mode)
  {
  }

  ~Renderer() noexcept(false)
  {
    if (m_gas_output_buffer) { m_gas_output_buffer.reset(); }

    if (m_raygen_records) { m_raygen_records.reset(); }
    if (m_miss_records) { m_miss_records.reset(); }
    if (m_hit_group_records) { m_hit_group_records.reset(); }

    if (m_pipeline) { OPTIX_CHECK(optixPipelineDestroy(m_pipeline)); }

    if (m_raygen_prog_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_raygen_prog_group));
    }
    if (m_miss_prog_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_miss_prog_group));
    }
    if (m_hit_prog_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_hit_prog_group));
    }

    if (m_module) { OPTIX_CHECK(optixModuleDestroy(m_module)); }

    if (m_context) { OPTIX_CHECK(optixDeviceContextDestroy(m_context)); }
  }

  void create_context()
  {
    CUDA_CHECK(cudaFree(0));

    CUcontext cu_cxt = 0;
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.validationMode = m_enable_validation_mode
                                 ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
                                 : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    options.logCallbackFunction = &context_log_callback;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_cxt, &options, &m_context));
  }

  void create_module(const std::filesystem::path& ptx_filepath)
  {
    OptixModuleCompileOptions options = {};
    options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    options.debugLevel = m_enable_validation_mode
                             ? OPTIX_COMPILE_DEBUG_LEVEL_FULL
                             : OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    options.optLevel = m_enable_validation_mode
                           ? OPTIX_COMPILE_OPTIMIZATION_LEVEL_0
                           : OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

    const std::vector<char> ptx = read_file(ptx_filepath);

    // TODO: move these outside this function as much as possible
    m_pipeline_compile_options.usesMotionBlur = false;
    m_pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    m_pipeline_compile_options.numPayloadValues = 3;
    m_pipeline_compile_options.numAttributeValues = 3;
    m_pipeline_compile_options.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    if (m_enable_validation_mode) {
      m_pipeline_compile_options.exceptionFlags =
          OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
          OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    } else {
      m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        m_context, &options, &m_pipeline_compile_options, ptx.data(),
        ptx.size(), log, &sizeof_log, &m_module));
  }

  // TODO: take module and entry function name from outside?
  void create_program_group()
  {
    OptixProgramGroupOptions options = {};

    // create raygen program group
    OptixProgramGroupDesc raygen_program_group_desc = {};
    raygen_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_program_group_desc.raygen.module = m_module;
    raygen_program_group_desc.raygen.entryFunctionName = "__raygen__rg";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &raygen_program_group_desc, 1, &options, log, &sizeof_log,
        &m_raygen_prog_group));

    // create miss program group
    OptixProgramGroupDesc miss_program_group_desc = {};
    miss_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_program_group_desc.miss.module = m_module;
    miss_program_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, &miss_program_group_desc,
                                            1, &options, log, &sizeof_log,
                                            &m_miss_prog_group));

    // create hitgroup program group
    OptixProgramGroupDesc hitgroup_program_group_desc = {};
    hitgroup_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_program_group_desc.hitgroup.moduleCH = m_module;
    hitgroup_program_group_desc.hitgroup.entryFunctionNameCH =
        "__closesthit__ch";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(m_context, &hitgroup_program_group_desc, 1,
                                &options, log, &sizeof_log, &m_hit_prog_group));
  }

  void create_pipeline(uint32_t max_trace_depth, uint32_t max_traversable_depth)
  {
    OptixProgramGroup program_groups[] = {m_raygen_prog_group,
                                          m_miss_prog_group, m_hit_prog_group};

    // create pipeline
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel = m_enable_validation_mode
                                           ? OPTIX_COMPILE_DEBUG_LEVEL_FULL
                                           : OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_context, &m_pipeline_compile_options, &pipeline_link_options,
        program_groups, sizeof(program_groups) / sizeof(program_groups[0]), log,
        &sizeof_log, &m_pipeline));

    // set pipeline stack size
    OptixStackSizes stack_sizes = {};
    for (auto& program_group : program_groups) {
      OPTIX_CHECK(optixUtilAccumulateStackSizes(program_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes, max_trace_depth, 0, 0,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));

    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        max_traversable_depth));
  }

  void create_sbt(const RayGenSbtRecord& raygen_sbt_record,
                  const std::vector<MissSbtRecord>& miss_sbt_records,
                  const std::vector<HitGroupSbtRecord>& hit_group_sbt_records)
  {
    // allocate records on device
    m_raygen_records = std::make_unique<DeviceBuffer<RayGenSbtRecord>>(1);
    m_miss_records =
        std::make_unique<DeviceBuffer<MissSbtRecord>>(miss_sbt_records.size());
    m_hit_group_records = std::make_unique<DeviceBuffer<HitGroupSbtRecord>>(
        hit_group_sbt_records.size());

    // copy raygen records to device
    OPTIX_CHECK(
        optixSbtRecordPackHeader(m_raygen_prog_group, &raygen_sbt_record));
    std::vector<RayGenSbtRecord> raygen_records_host = {raygen_sbt_record};
    m_raygen_records->copy_from_host_to_device(raygen_records_host);

    // copy miss records to device
    for (auto& record : miss_sbt_records) {
      OPTIX_CHECK(optixSbtRecordPackHeader(m_miss_prog_group, &record));
    }
    m_miss_records->copy_from_host_to_device(miss_sbt_records);

    // copy hit group records to device
    for (auto& record : hit_group_sbt_records) {
      OPTIX_CHECK(optixSbtRecordPackHeader(m_hit_prog_group, &record));
    }
    m_hit_group_records->copy_from_host_to_device(hit_group_sbt_records);

    // fill sbt
    m_sbt.raygenRecord =
        reinterpret_cast<CUdeviceptr>(m_raygen_records->get_device_ptr());

    m_sbt.missRecordBase =
        reinterpret_cast<CUdeviceptr>(m_miss_records->get_device_ptr());
    m_sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    m_sbt.missRecordCount = miss_sbt_records.size();

    m_sbt.hitgroupRecordBase =
        reinterpret_cast<CUdeviceptr>(m_hit_group_records->get_device_ptr());
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    m_sbt.hitgroupRecordCount = hit_group_sbt_records.size();
  }

  void load_scene(const Scene& scene)
  {
    // GAS build option
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // GAS input
    // TODO: use indices
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.numVertices = scene.get_vertices_size();
    input.triangleArray.vertexStrideInBytes = sizeof(float3);
    CUdeviceptr d_vertices =
        reinterpret_cast<CUdeviceptr>(scene.get_vertices_device_ptr());
    input.triangleArray.vertexBuffers = &d_vertices;

    // TODO: set SBT records
    const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    input.triangleArray.flags = flags;
    input.triangleArray.numSbtRecords = 1;

    // compute GAS buffer size
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &options, &input, 1,
                                             &gas_buffer_sizes));

    // build GAS
    DeviceBuffer<uint8_t> gas_temp_buffer(gas_buffer_sizes.tempSizeInBytes);
    m_gas_output_buffer = std::make_unique<DeviceBuffer<uint8_t>>(
        gas_buffer_sizes.outputSizeInBytes);
    OPTIX_CHECK(optixAccelBuild(
        m_context, 0, &options, &input, 1,
        reinterpret_cast<CUdeviceptr>(gas_temp_buffer.get_device_ptr()),
        gas_buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(m_gas_output_buffer->get_device_ptr()),
        gas_buffer_sizes.outputSizeInBytes, &m_gas_handle, nullptr, 0));
  }

 private:
  uint32_t m_width = 0;
  uint32_t m_height = 0;
  bool m_enable_validation_mode = false;

  OptixDeviceContext m_context = 0;

  OptixTraversableHandle m_gas_handle = 0;
  std::unique_ptr<DeviceBuffer<uint8_t>> m_gas_output_buffer = nullptr;

  OptixModule m_module = 0;

  OptixProgramGroup m_raygen_prog_group = 0;
  OptixProgramGroup m_miss_prog_group = 0;
  OptixProgramGroup m_hit_prog_group = 0;

  OptixPipelineCompileOptions m_pipeline_compile_options = {};
  OptixPipeline m_pipeline = 0;

  std::unique_ptr<DeviceBuffer<RayGenSbtRecord>> m_raygen_records = nullptr;
  std::unique_ptr<DeviceBuffer<MissSbtRecord>> m_miss_records = nullptr;
  std::unique_ptr<DeviceBuffer<HitGroupSbtRecord>> m_hit_group_records =
      nullptr;
  OptixShaderBindingTable m_sbt = {};

  static void context_log_callback(unsigned int level, const char* tag,
                                   const char* message, void* cbdata)
  {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
              << "]: " << message << std::endl;
  }

  static std::vector<char> read_file(const std::filesystem::path& filepath)
  {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("failed to open " + filepath.generic_string());
    }

    const size_t file_size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(file_size);

    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();

    return buffer;
  }
};