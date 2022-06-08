#include <cuda_runtime.h>
#include <optix.h>

#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "optix_stubs.h"

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

class App
{
 public:
  App() {}

  void init() { create_context(); }
  void render();
  void cleanup() { OPTIX_CHECK(optixDeviceContextDestroy(context)); }

 private:
  OptixDeviceContext context;

  void create_context()
  {
    CUDA_CHECK(cudaFree(0));

    CUcontext cu_cxt = 0;
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_callback;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_cxt, &options, &context));
  }

  static void context_log_callback(unsigned int level, const char* tag,
                                   const char* message, void* cbdata)
  {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
              << "]: " << message << std::endl;
  }
};

int main() { return 0; }