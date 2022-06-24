#pragma once

#include <optix.h>
#include <optix_stubs.h>

#include <sstream>
#include <stdexcept>

#include "spdlog/spdlog.h"

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

namespace optwl
{

// RAII wrapper for Optix Context
struct Context {
  OptixDeviceContext m_context = {};

  Context(CUcontext cu_cxt = 0)
  {
#ifdef NDEBUG
    bool enable_validation_mode = false;
#else
    bool enable_validation_mode = true;
#endif

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.validationMode = enable_validation_mode
                                 ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
                                 : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    options.logCallbackFunction = &log_callback;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_cxt, &options, &m_context));
  }

  ~Context() noexcept(false)
  {
    if (m_context) { OPTIX_CHECK(optixDeviceContextDestroy(m_context)); }
  }

  static void log_callback(unsigned int level, const char* tag,
                           const char* message, void* cbdata)
  {
    if (level == 4) {
      spdlog::info("[OptiX][{}] {}", tag, message);
    } else if (level == 3) {
      spdlog::warn("[OptiX][{}] {}", tag, message);
    } else if (level == 2) {
      spdlog::error("[OptiX][{}] {}", tag, message);
    } else if (level == 1) {
      spdlog::critical("[OptiX][{}] {}", tag, message);
    }
  }
};

}  // namespace optwl