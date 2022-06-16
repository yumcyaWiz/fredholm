#pragma once
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "camera.h"
#include "device/buffer.h"
#include "device/texture.h"
#include "device/util.h"
#include "io.h"
#include "scene.h"
#include "shared.h"

namespace fredholm
{

class Renderer
{
 public:
  Renderer(uint32_t width, uint32_t height, bool enable_validation_mode = false)
      : m_width(width),
        m_height(height),
        m_enable_validation_mode(enable_validation_mode),
        m_framebuffer(width, height),
        m_accumulation(width, height),
        m_sample_count(width, height)
  {
    CUDA_CHECK(cudaStreamCreate(&m_stream));
  }

  ~Renderer() noexcept(false)
  {
    // release scene data
    if (m_vertices) { m_vertices.reset(); }

    // release GAS
    if (m_gas_output_buffer) { m_gas_output_buffer.reset(); }

    // release SBT records
    if (m_raygen_records_d) { m_raygen_records_d.reset(); }
    if (m_miss_records_d) { m_miss_records_d.reset(); }
    if (m_hit_group_records_d) { m_hit_group_records_d.reset(); }

    // release pipeline
    if (m_pipeline) { OPTIX_CHECK(optixPipelineDestroy(m_pipeline)); }

    // release program groups
    if (m_raygen_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_raygen_group));
    }
    if (m_radiance_miss_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_radiance_miss_group));
    }
    if (m_shadow_miss_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_shadow_miss_group));
    }
    if (m_radiance_hit_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_radiance_hit_group));
    }
    if (m_shadow_hit_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_shadow_hit_group));
    }

    // release module
    if (m_module) { OPTIX_CHECK(optixModuleDestroy(m_module)); }

    // release context
    if (m_context) { OPTIX_CHECK(optixDeviceContextDestroy(m_context)); }

    CUDA_CHECK(cudaStreamDestroy(m_stream));
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
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(m_context, &raygen_program_group_desc, 1,
                                &options, log, &sizeof_log, &m_raygen_group));

    // create miss program group
    OptixProgramGroupDesc miss_program_group_desc = {};
    miss_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_program_group_desc.miss.module = m_module;
    miss_program_group_desc.miss.entryFunctionName = "__miss__radiance";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, &miss_program_group_desc,
                                            1, &options, log, &sizeof_log,
                                            &m_radiance_miss_group));

    miss_program_group_desc.miss.entryFunctionName = "__miss__shadow";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, &miss_program_group_desc,
                                            1, &options, log, &sizeof_log,
                                            &m_shadow_miss_group));

    // create hitgroup program group
    OptixProgramGroupDesc hitgroup_program_group_desc = {};
    hitgroup_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_program_group_desc.hitgroup.moduleCH = m_module;
    hitgroup_program_group_desc.hitgroup.entryFunctionNameCH =
        "__closesthit__radiance";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &hitgroup_program_group_desc, 1, &options, log, &sizeof_log,
        &m_radiance_hit_group));

    hitgroup_program_group_desc.hitgroup.entryFunctionNameCH =
        "__closesthit__shadow";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &hitgroup_program_group_desc, 1, &options, log, &sizeof_log,
        &m_shadow_hit_group));
  }

  void create_pipeline()
  {
    OptixProgramGroup program_groups[] = {
        m_raygen_group, m_radiance_miss_group, m_shadow_miss_group,
        m_radiance_hit_group, m_shadow_hit_group};

    // create pipeline
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = m_max_trace_depth;
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
        &stack_sizes, m_max_trace_depth, 0, 0,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));

    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        m_max_traversable_depth));
  }

  void create_sbt(const Scene& scene)
  {
    // fill raygen header
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_group, &m_raygen_record));

    // fill miss record
    MissSbtRecord miss_record = {};
    miss_record.data.bg_color = make_float3(0.0f);
    OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_miss_group, &miss_record));
    m_miss_records.push_back(miss_record);

    miss_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_shadow_miss_group, &miss_record));
    m_miss_records.push_back(miss_record);

    // fill hitgroup record
    // TODO: use per-material GAS and single IAS, to reduce the number of SBT
    // records
    // TODO: move this inside load_scene?
    for (size_t f = 0; f < scene.n_faces(); ++f) {
      const uint material_id = scene.m_material_ids[f];
      const Material& material = scene.m_materials[material_id];

      // radiance hitgroup record
      HitGroupSbtRecord hit_record = {};
      hit_record.data.vertices = m_vertices->get_device_ptr() + 3 * f;
      hit_record.data.material = scene.m_materials[material_id];
      OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_hit_group, &hit_record));
      m_hit_group_records.push_back(hit_record);

      // shadow hitgroup record
      hit_record = {};
      OPTIX_CHECK(optixSbtRecordPackHeader(m_shadow_hit_group, &hit_record));
      m_hit_group_records.push_back(hit_record);
    }

    // allocate SBT records on device
    std::vector<RayGenSbtRecord> raygen_sbt_records = {m_raygen_record};
    m_raygen_records_d =
        std::make_unique<DeviceBuffer<RayGenSbtRecord>>(raygen_sbt_records);
    m_miss_records_d =
        std::make_unique<DeviceBuffer<MissSbtRecord>>(m_miss_records);
    m_hit_group_records_d =
        std::make_unique<DeviceBuffer<HitGroupSbtRecord>>(m_hit_group_records);

    // fill SBT
    m_sbt.raygenRecord =
        reinterpret_cast<CUdeviceptr>(m_raygen_records_d->get_device_ptr());

    m_sbt.missRecordBase =
        reinterpret_cast<CUdeviceptr>(m_miss_records_d->get_device_ptr());
    m_sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    m_sbt.missRecordCount = m_miss_records.size();

    m_sbt.hitgroupRecordBase =
        reinterpret_cast<CUdeviceptr>(m_hit_group_records_d->get_device_ptr());
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    m_sbt.hitgroupRecordCount = m_hit_group_records.size();
  }

  void load_scene(const Scene& scene)
  {
    if (!scene.is_valid()) { throw std::runtime_error("invalid scene"); }

    m_vertices = std::make_unique<DeviceBuffer<float3>>(scene.m_vertices);
  }

  void build_accel()
  {
    // GAS build option
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // GAS input
    // TODO: use indices
    uint32_t num_faces = m_vertices->get_size() / 3;
    std::vector<OptixBuildInput> inputs(num_faces);
    // NOTE: need this, since vertexBuffers take a pointer to array of device
    // pointers
    std::vector<CUdeviceptr> per_face_vertex_buffers(num_faces);

    for (int f = 0; f < num_faces; ++f) {
      OptixBuildInput input = {};
      input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      input.triangleArray.numVertices = 3;
      input.triangleArray.vertexStrideInBytes = sizeof(float3);

      per_face_vertex_buffers[f] =
          reinterpret_cast<CUdeviceptr>(m_vertices->get_device_ptr() + 3 * f);
      input.triangleArray.vertexBuffers = &per_face_vertex_buffers[f];

      const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
      input.triangleArray.flags = flags;

      input.triangleArray.numSbtRecords = 1;

      inputs[f] = input;
    }

    // compute GAS buffer size
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &options, inputs.data(),
                                             inputs.size(), &gas_buffer_sizes));

    // build GAS
    DeviceBuffer<uint8_t> gas_temp_buffer(gas_buffer_sizes.tempSizeInBytes);
    m_gas_output_buffer = std::make_unique<DeviceBuffer<uint8_t>>(
        gas_buffer_sizes.outputSizeInBytes);
    OPTIX_CHECK(optixAccelBuild(
        m_context, 0, &options, inputs.data(), inputs.size(),
        reinterpret_cast<CUdeviceptr>(gas_temp_buffer.get_device_ptr()),
        gas_buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(m_gas_output_buffer->get_device_ptr()),
        gas_buffer_sizes.outputSizeInBytes, &m_gas_handle, nullptr, 0));
  }

  void render(const Camera& camera, uint32_t n_samples, uint32_t max_depth)
  {
    LaunchParams params;
    params.framebuffer = m_framebuffer.get_device_ptr();
    params.width = m_width;
    params.height = m_height;
    params.n_samples = n_samples;
    params.max_depth = max_depth;

    params.cam_origin = camera.m_origin;
    params.cam_forward = camera.m_forward;
    params.cam_right = camera.m_right;
    params.cam_up = camera.m_up;

    params.gas_handle = m_gas_handle;

    DeviceObject d_params(params);

    // run pipeline
    OPTIX_CHECK(
        optixLaunch(m_pipeline, m_stream,
                    reinterpret_cast<CUdeviceptr>(d_params.get_device_ptr()),
                    sizeof(LaunchParams), &m_sbt, m_width, m_height, 1));
  }

  void init_rng_state()
  {
    // TODO: apply some hash function
    std::vector<RNGState> rng_states(m_width * m_height);
    for (int j = 0; j < m_height; ++j) {
      for (int i = 0; i < m_width; ++i) {
        const int idx = i + m_width * j;
        rng_states[idx].state = idx;
        rng_states[idx].inc = 0xdeadbeef;
      }
    }

    m_rng_states = std::make_unique<DeviceBuffer<RNGState>>(rng_states);
  }

  void render_one_sample(const Camera& camera, float4* d_framebuffer,
                         uint32_t max_depth)
  {
    LaunchParams params;
    params.framebuffer = d_framebuffer;
    params.accumulation = m_accumulation.get_device_ptr();
    params.sample_count = m_sample_count.get_device_ptr();
    params.rng_states = m_rng_states->get_device_ptr();

    params.width = m_width;
    params.height = m_height;
    params.n_samples = 1;
    params.max_depth = max_depth;

    params.cam_origin = camera.m_origin;
    params.cam_forward = camera.m_forward;
    params.cam_right = camera.m_right;
    params.cam_up = camera.m_up;

    params.gas_handle = m_gas_handle;

    DeviceObject d_params(params);

    // run pipeline
    OPTIX_CHECK(
        optixLaunch(m_pipeline, m_stream,
                    reinterpret_cast<CUdeviceptr>(d_params.get_device_ptr()),
                    sizeof(LaunchParams), &m_sbt, m_width, m_height, 1));
  }

  void wait_for_completion() { CUDA_SYNC_CHECK(); }

  void write_framebuffer_as_ppm(const std::filesystem::path& filepath)
  {
    m_framebuffer.copy_from_device_to_host();
    write_ppm(m_framebuffer, filepath);
  }

 private:
  uint32_t m_width = 0;
  uint32_t m_height = 0;

  bool m_enable_validation_mode = false;

  uint32_t m_max_traversable_depth = 1;
  uint32_t m_max_trace_depth = 3;

  std::unique_ptr<DeviceBuffer<float3>> m_vertices = nullptr;

  CUstream m_stream = 0;

  OptixDeviceContext m_context = 0;

  OptixTraversableHandle m_gas_handle = 0;
  std::unique_ptr<DeviceBuffer<uint8_t>> m_gas_output_buffer = nullptr;

  OptixModule m_module = 0;

  OptixProgramGroup m_raygen_group = 0;
  OptixProgramGroup m_radiance_miss_group = 0;
  OptixProgramGroup m_shadow_miss_group = 0;
  OptixProgramGroup m_radiance_hit_group = 0;
  OptixProgramGroup m_shadow_hit_group = 0;

  OptixPipelineCompileOptions m_pipeline_compile_options = {};
  OptixPipeline m_pipeline = 0;

  RayGenSbtRecord m_raygen_record = {};
  std::vector<MissSbtRecord> m_miss_records = {};
  std::vector<HitGroupSbtRecord> m_hit_group_records = {};
  std::unique_ptr<DeviceBuffer<RayGenSbtRecord>> m_raygen_records_d = nullptr;
  std::unique_ptr<DeviceBuffer<MissSbtRecord>> m_miss_records_d = nullptr;
  std::unique_ptr<DeviceBuffer<HitGroupSbtRecord>> m_hit_group_records_d =
      nullptr;
  OptixShaderBindingTable m_sbt = {};

  // TODO: use device buffer
  Texture2D<float4> m_framebuffer;
  Texture2D<float4> m_accumulation;
  Texture2D<uint> m_sample_count;
  std::unique_ptr<DeviceBuffer<RNGState>> m_rng_states;

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

}  // namespace fredholm