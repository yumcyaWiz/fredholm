#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "device/texture.h"
#include "device/util.h"
#include "io.h"
#include "scene.h"
#include "shared.h"

std::vector<char> read_file(const std::filesystem::path& filepath)
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

template <typename T>
struct SbtRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

using RayGenSbtRecord = SbtRecord<int>;
using MissSbtRecord = SbtRecord<int>;
using HitGroupSbtRecord = SbtRecord<int>;

class App
{
 public:
  App(uint32_t width, uint32_t height) : m_width(width), m_height(height)
  {
#ifdef NDEBUG
    enable_validation_mode = false;
#else
    m_enable_validation_mode = true;
#endif
  }

  void init()
  {
    create_context();
    create_accel_structure();
    init_pipeline_compile_options();
    create_module(std::filesystem::path(MODULES_SOURCE_DIR) / "triangle.ptx");
    create_program_group();
    create_pipeline();
    create_shader_binding_table();
  }

  void render()
  {
    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    Texture2D<float4> image(m_width, m_height);

    Params params;
    params.image = image.get_device_ptr();
    params.image_width = m_width;
    params.image_height = m_height;
    params.cam_origin = make_float3(0.0f, 0.0f, 1.0f);
    params.cam_forward = make_float3(0.0f, 0.0f, -1.0f);
    params.cam_right = make_float3(1.0f, 0.0f, 0.0f);
    params.cam_up = make_float3(0.0f, 1.0f, 0.0f);
    params.handle = m_gas_handle;

    DeviceObject d_params(params);

    // run pipeline
    OPTIX_CHECK(
        optixLaunch(m_pipeline, stream,
                    reinterpret_cast<CUdeviceptr>(d_params.get_device_ptr()),
                    sizeof(Params), &m_sbt, m_width, m_height, 1));
    CUDA_SYNC_CHECK();

    // save image as ppm
    image.copy_from_device_to_host();
    write_ppm(image, "output.ppm");

    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void cleanup()
  {
    m_raygen_records.reset();
    m_miss_records.reset();
    m_hitgroup_records.reset();

    OPTIX_CHECK(optixPipelineDestroy(m_pipeline));

    OPTIX_CHECK(optixProgramGroupDestroy(m_raygen_program_group));
    OPTIX_CHECK(optixProgramGroupDestroy(m_miss_program_group));
    OPTIX_CHECK(optixProgramGroupDestroy(m_hitgroup_program_group));

    OPTIX_CHECK(optixModuleDestroy(m_module));

    m_gas_output_buffer.reset();

    OPTIX_CHECK(optixDeviceContextDestroy(m_context));
  }

 private:
  uint32_t m_width = 0;
  uint32_t m_height = 0;
  bool m_enable_validation_mode = false;

  OptixDeviceContext m_context = nullptr;

  OptixTraversableHandle m_gas_handle;
  std::unique_ptr<DeviceBuffer<uint8_t>> m_gas_output_buffer = nullptr;

  OptixPipelineCompileOptions m_pipeline_compile_options = {};
  OptixModule m_module = nullptr;

  OptixProgramGroup m_raygen_program_group = nullptr;
  OptixProgramGroup m_miss_program_group = nullptr;
  OptixProgramGroup m_hitgroup_program_group = nullptr;

  OptixPipeline m_pipeline = nullptr;

  OptixShaderBindingTable m_sbt = {};
  std::unique_ptr<Buffer<RayGenSbtRecord>> m_raygen_records = nullptr;
  std::unique_ptr<Buffer<MissSbtRecord>> m_miss_records = nullptr;
  std::unique_ptr<Buffer<HitGroupSbtRecord>> m_hitgroup_records = nullptr;

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

  void create_accel_structure()
  {
    // GAS build option
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Vertex buffer
    Buffer<float3> vertices(3);
    vertices.set_value(0, {-0.5f, -0.5f, 0.0f});
    vertices.set_value(1, {0.5f, -0.5f, 0.0f});
    vertices.set_value(2, {0.0f, 0.5f, 0.0f});
    vertices.copy_from_host_to_device();

    // GAS input
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.numVertices = vertices.get_size();
    CUdeviceptr d_vertices =
        reinterpret_cast<CUdeviceptr>(vertices.get_device_ptr());
    input.triangleArray.vertexBuffers = &d_vertices;
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

  void init_pipeline_compile_options()
  {
    m_pipeline_compile_options.usesMotionBlur = false;

    m_pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

    m_pipeline_compile_options.numPayloadValues = 3;
    m_pipeline_compile_options.numAttributeValues = 3;

    if (m_enable_validation_mode) {
      m_pipeline_compile_options.exceptionFlags =
          OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
          OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    } else {
      m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    }

    m_pipeline_compile_options.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
  }

  void create_module(const std::filesystem::path& ptx_filepath)
  {
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    if (m_enable_validation_mode) {
      module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
      module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    } else {
      module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
      module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    }

    const std::vector<char> ptx = read_file(ptx_filepath);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        m_context, &module_compile_options, &m_pipeline_compile_options,
        ptx.data(), ptx.size(), log, &sizeof_log, &m_module));
  }

  void create_program_group()
  {
    OptixProgramGroupOptions program_group_options = {};

    // create raygen program group
    OptixProgramGroupDesc raygen_program_group_desc = {};
    raygen_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_program_group_desc.raygen.module = m_module;
    raygen_program_group_desc.raygen.entryFunctionName = "__raygen__rg";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &raygen_program_group_desc, 1, &program_group_options, log,
        &sizeof_log, &m_raygen_program_group));

    // create miss program group
    OptixProgramGroupDesc miss_program_group_desc = {};
    miss_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_program_group_desc.miss.module = m_module;
    miss_program_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &miss_program_group_desc, 1, &program_group_options, log,
        &sizeof_log, &m_miss_program_group));

    // create hitgroup program group
    OptixProgramGroupDesc hitgroup_program_group_desc = {};
    hitgroup_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_program_group_desc.hitgroup.moduleCH = m_module;
    hitgroup_program_group_desc.hitgroup.entryFunctionNameCH =
        "__closesthit__ch";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &hitgroup_program_group_desc, 1, &program_group_options, log,
        &sizeof_log, &m_hitgroup_program_group));
  }

  void create_pipeline()
  {
    const uint32_t max_trace_depth = 1;
    const uint32_t max_traversable_depth = 1;

    OptixProgramGroup program_groups[] = {
        m_raygen_program_group, m_miss_program_group, m_hitgroup_program_group};

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

  void create_shader_binding_table()
  {
    m_raygen_records = std::make_unique<Buffer<RayGenSbtRecord>>(1);
    m_miss_records = std::make_unique<Buffer<MissSbtRecord>>(1);
    m_hitgroup_records = std::make_unique<Buffer<HitGroupSbtRecord>>(1);

    RayGenSbtRecord raygen_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_program_group, &raygen_sbt));
    m_raygen_records->set_value(0, raygen_sbt);
    m_raygen_records->copy_from_host_to_device();

    MissSbtRecord miss_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_miss_program_group, &miss_sbt));
    m_miss_records->set_value(0, miss_sbt);
    m_miss_records->copy_from_host_to_device();

    HitGroupSbtRecord hitgroup_sbt;
    OPTIX_CHECK(
        optixSbtRecordPackHeader(m_hitgroup_program_group, &hitgroup_sbt));
    m_hitgroup_records->set_value(0, hitgroup_sbt);
    m_hitgroup_records->copy_from_host_to_device();

    m_sbt.raygenRecord =
        reinterpret_cast<CUdeviceptr>(m_raygen_records->get_device_ptr());

    m_sbt.missRecordBase =
        reinterpret_cast<CUdeviceptr>(m_miss_records->get_device_ptr());
    m_sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    m_sbt.missRecordCount = 1;

    m_sbt.hitgroupRecordBase =
        reinterpret_cast<CUdeviceptr>(m_hitgroup_records->get_device_ptr());
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    m_sbt.hitgroupRecordCount = 1;
  }

  static void context_log_callback(unsigned int level, const char* tag,
                                   const char* message, void* cbdata)
  {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
              << "]: " << message << std::endl;
  }
};

int main()
{
  const uint32_t width = 512;
  const uint32_t height = 512;

  try {
    App app(width, height);
    app.init();

    app.render();

    app.cleanup();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}