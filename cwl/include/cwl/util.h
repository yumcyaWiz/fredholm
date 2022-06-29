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

namespace cwl
{

// RAII wrapper for objects on device
template <typename T>
class DeviceObject
{
 public:
  DeviceObject(const T& object)
  {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_device_ptr), sizeof(T)));

    CUDA_CHECK(
        cudaMemcpy(m_device_ptr, &object, sizeof(T), cudaMemcpyHostToDevice));
  }

  ~DeviceObject() noexcept(false)
  {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_ptr)));
  }

  T* get_device_ptr() const { return m_device_ptr; }

 private:
  T* m_device_ptr;
};

}  // namespace cwl