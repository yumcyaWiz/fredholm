#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
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
  }

  void render();

  void cleanup()
  {
    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));
  }

 private:
  bool enable_validation_mode = false;

  OptixDeviceContext context;

  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixModule module;

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
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    std::vector<char> ptx = read_file(ptx_filepath);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        context, &module_compile_options, &pipeline_compile_options, ptx.data(),
        ptx.size(), log, &sizeof_log, &module));
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