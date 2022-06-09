#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include <sstream>
#include <stdexcept>

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

#define CUDA_SYNC_CHECK()                                                  \
  do {                                                                     \
    cudaDeviceSynchronize();                                               \
    cudaError_t error = cudaGetLastError();                                \
    if (error != cudaSuccess) {                                            \
      std::stringstream ss;                                                \
      ss << "CUDA error on synchronize with error '"                       \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__ \
         << ")\n";                                                         \
      throw std::runtime_error(ss.str().c_str());                          \
    }                                                                      \
  } while (0)

template <typename T>
inline CUdeviceptr alloc_and_copy_to_device(const T& src)
{
  // alloc memory on device
  CUdeviceptr dst;
  const size_t size = sizeof(T);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), size));

  // copy data from host to device
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dst), &src, size,
                        cudaMemcpyHostToDevice));

  return dst;
}