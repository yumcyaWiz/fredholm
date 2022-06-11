
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

class App
{
 public:
  App(uint32_t width, uint32_t height) : width(width), height(height)
  {
#ifdef NDEBUG
    enable_validation_mode = false;
#else
    enable_validation_mode = true;
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

    Texture2D<float4> image(width, height);

    Params params;
    params.image = image.get_device_ptr();
    params.image_width = width;
    params.image_height = height;

    DeviceObject d_params(params);

    // run pipeline
    OPTIX_CHECK(
        optixLaunch(pipeline, stream,
                    reinterpret_cast<CUdeviceptr>(d_params.get_device_ptr()),
                    sizeof(Params), &sbt, width, height, 1));
    CUDA_SYNC_CHECK();

    // save image as ppm
    image.copy_from_device_to_host();
    write_ppm(image, "output.ppm");

    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void cleanup()
  {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));

    OPTIX_CHECK(optixPipelineDestroy(pipeline));

    OPTIX_CHECK(optixProgramGroupDestroy(raygen_program_group));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_program_group));

    OPTIX_CHECK(optixModuleDestroy(module));

    gas_output_buffer.reset();

    OPTIX_CHECK(optixDeviceContextDestroy(context));
  }

 private:
  uint32_t width = 0;
  uint32_t height = 0;
  bool enable_validation_mode = false;

  OptixDeviceContext context = nullptr;

  OptixTraversableHandle gas_handle;
  std::unique_ptr<DeviceBuffer<uint8_t>> gas_output_buffer = nullptr;

  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixModule module = nullptr;

  OptixProgramGroup raygen_program_group = nullptr;
  OptixProgramGroup miss_program_group = nullptr;
  OptixProgramGroup hitgroup_program_group = nullptr;

  OptixPipeline pipeline = nullptr;

  OptixShaderBindingTable sbt = {};

  void create_context()
  {
    CUDA_CHECK(cudaFree(0));

    CUcontext cu_cxt = 0;
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.validationMode = enable_validation_mode
                                 ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
                                 : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    options.logCallbackFunction = &context_log_callback;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_cxt, &options, &context));
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
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &options, &input, 1,
                                             &gas_buffer_sizes));

    // build GAS
    DeviceBuffer<uint8_t> gas_temp_buffer(gas_buffer_sizes.tempSizeInBytes);
    gas_output_buffer = std::make_unique<DeviceBuffer<uint8_t>>(
        gas_buffer_sizes.outputSizeInBytes);
    OPTIX_CHECK(optixAccelBuild(
        context, 0, &options, &input, 1,
        reinterpret_cast<CUdeviceptr>(gas_temp_buffer.get_device_ptr()),
        gas_buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(gas_output_buffer->get_device_ptr()),
        gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));
  }

  void init_pipeline_compile_options()
  {
    pipeline_compile_options.usesMotionBlur = false;

    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;

    pipeline_compile_options.numPayloadValues = 3;
    pipeline_compile_options.numAttributeValues = 3;

    if (enable_validation_mode) {
      pipeline_compile_options.exceptionFlags =
          OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
          OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    } else {
      pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    }

    pipeline_compile_options.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
  }

  void create_module(const std::filesystem::path& ptx_filepath)
  {
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    if (enable_validation_mode) {
      module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
      module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    } else {
      module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
      module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    }

    std::vector<char> ptx = read_file(ptx_filepath);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        context, &module_compile_options, &pipeline_compile_options, ptx.data(),
        ptx.size(), log, &sizeof_log, &module));
  }

  void create_program_group()
  {
    OptixProgramGroupOptions program_group_options = {};

    // create raygen program group
    OptixProgramGroupDesc raygen_program_group_desc = {};
    raygen_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_program_group_desc.raygen.module = module;
    raygen_program_group_desc.raygen.entryFunctionName = "__raygen__rg";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context, &raygen_program_group_desc, 1, &program_group_options, log,
        &sizeof_log, &raygen_program_group));

    // create miss program group
    OptixProgramGroupDesc miss_program_group_desc = {};
    miss_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_program_group_desc.miss.module = module;
    miss_program_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_program_group_desc,
                                            1, &program_group_options, log,
                                            &sizeof_log, &miss_program_group));

    // create hitgroup program group
    OptixProgramGroupDesc hitgroup_program_group_desc = {};
    hitgroup_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_program_group_desc.hitgroup.moduleCH = module;
    hitgroup_program_group_desc.hitgroup.entryFunctionNameCH =
        "__closesthit__ch";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context, &hitgroup_program_group_desc, 1, &program_group_options, log,
        &sizeof_log, &hitgroup_program_group));
  }

  void create_pipeline()
  {
    const uint32_t max_trace_depth = 0;

    OptixProgramGroup program_groups[] = {raygen_program_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel = enable_validation_mode
                                           ? OPTIX_COMPILE_DEBUG_LEVEL_FULL
                                           : OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        context, &pipeline_compile_options, &pipeline_link_options,
        program_groups, sizeof(program_groups) / sizeof(program_groups[0]), log,
        &sizeof_log, &pipeline));

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
        pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size, 2));
  }

  void create_shader_binding_table()
  {
    RayGenSbtRecord raygen_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_program_group, &raygen_sbt));
    CUdeviceptr raygen_record = alloc_and_copy_to_device(raygen_sbt);

    MissSbtRecord miss_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_program_group, &miss_sbt));
    CUdeviceptr miss_record = alloc_and_copy_to_device(miss_sbt);

    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;
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