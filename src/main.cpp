#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
      std::stringstream ss;                                                \
      ss << "CUDA call (" << #call << " ) failed with error: '"            \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__ \
         << ")\n";                                                         \
      throw std::runtime_error(ss.str().c_str());                          \
    }                                                                      \
  } while (0)

#define OPTIX_CHECK(call)                                                    \
  do {                                                                       \
    OptixResult res = call;                                                  \
    if (res != OPTIX_SUCCESS) {                                              \
      std::stringstream ss;                                                  \
      ss << "Optix call '" << #call << "' failed: " __FILE__ ":" << __LINE__ \
         << ")\n";                                                           \
      throw std::runtime_error(ss.str().c_str());                            \
    }                                                                        \
  } while (0)

#define OPTIX_CHECK_LOG(call)                                                \
  do {                                                                       \
    OptixResult res = call;                                                  \
    const size_t sizeof_log_returned = sizeof_log;                           \
    sizeof_log = sizeof(log); /* reset sizeof_log for future calls */        \
    if (res != OPTIX_SUCCESS) {                                              \
      std::stringstream ss;                                                  \
      ss << "Optix call '" << #call << "' failed: " __FILE__ ":" << __LINE__ \
         << ")\nLog:\n"                                                      \
         << log << (sizeof_log_returned > sizeof(log) ? "<TRUNCATED>" : "")  \
         << "\n";                                                            \
      throw std::runtime_error(ss.str().c_str());                            \
    }                                                                        \
  } while (0)

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

class App
{
 public:
  App()
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
    init_pipeline_compile_options();
    create_module(std::filesystem::path(MODULES_SOURCE_DIR) / "white.ptx");
    create_program_group();
    create_pipeline();
  }

  void render();

  void cleanup()
  {
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_program_group));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_program_group));
    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));
  }

 private:
  bool enable_validation_mode = false;

  OptixDeviceContext context = nullptr;

  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixModule module = nullptr;

  OptixProgramGroup raygen_program_group = nullptr;
  OptixProgramGroup miss_program_group = nullptr;

  OptixPipeline pipeline = nullptr;

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

  void init_pipeline_compile_options()
  {
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
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

    OptixProgramGroupDesc raygen_program_group_desc = {};
    raygen_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_program_group_desc.raygen.module = module;
    raygen_program_group_desc.raygen.entryFunctionName = "__raygen__white";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context, &raygen_program_group_desc, 1, &program_group_options, log,
        &sizeof_log, &raygen_program_group));

    OptixProgramGroupDesc miss_program_group_desc = {};
    miss_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_program_group_desc,
                                            1, &program_group_options, log,
                                            &sizeof_log, &miss_program_group));
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

  static void context_log_callback(unsigned int level, const char* tag,
                                   const char* message, void* cbdata)
  {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
              << "]: " << message << std::endl;
  }
};

int main()
{
  App app;
  app.init();

  app.cleanup();

  return 0;
}