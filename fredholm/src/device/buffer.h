#pragma once
#include <cstring>
#include <vector>

#include "device/types.h"
#include "device/util.h"

namespace fredholm
{

// RAII buffer object which is on device
template <typename T>
class DeviceBuffer
{
 public:
  DeviceBuffer(uint32_t buffer_size) : m_buffer_size(buffer_size)
  {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_device_ptr),
                          m_buffer_size * sizeof(T)));
  }

  DeviceBuffer(const std::vector<T>& values) : DeviceBuffer<T>(values.size())
  {
    copy_from_host_to_device(values);
  }

  ~DeviceBuffer() noexcept(false)
  {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_ptr)));
  }

  void copy_from_host_to_device(const std::vector<T>& value)
  {
    CUDA_CHECK(cudaMemcpy(m_device_ptr, value.data(), m_buffer_size * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  T* get_device_ptr() const { return m_device_ptr; }

  uint32_t get_size() const { return m_buffer_size; }

  uint32_t get_size_in_bytes() const { return m_buffer_size * sizeof(T); }

 private:
  uint32_t m_buffer_size;
  T* m_device_ptr;
};

// RAII buffer object which is both on host and device
template <typename T>
class Buffer : public DeviceAndHostObject<T>
{
 public:
  Buffer(uint32_t buffer_size) : m_buffer_size(buffer_size)
  {
    // allocate on host
    m_host_ptr = reinterpret_cast<T*>(malloc(m_buffer_size * sizeof(T)));
    memset(m_host_ptr, 0, m_buffer_size * sizeof(T));

    // allocate on device
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_device_ptr),
                          m_buffer_size * sizeof(T)));
    CUDA_CHECK(cudaMemset(m_device_ptr, 0, m_buffer_size * sizeof(T)));
  }

  Buffer(const std::vector<T>& values) : Buffer(values.size())
  {
    // copy values to host buffer
    memcpy(m_host_ptr, values.data(), values.size() * sizeof(T));

    copy_from_host_to_device();
  }

  ~Buffer() noexcept(false)
  {
    // free memory on host
    free(m_host_ptr);

    // free memory on device
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_ptr)));
  }

  uint32_t get_size() const { return m_buffer_size; }

  T get_value(uint32_t index) const { return m_host_ptr[index]; }

  void set_value(uint32_t index, const T& value) { m_host_ptr[index] = value; }

  T* get_host_ptr() const override { return m_host_ptr; }

  T* get_device_ptr() const override { return m_device_ptr; }

  uint32_t get_size_in_bytes() const override
  {
    return m_buffer_size * sizeof(T);
  }

  void copy_from_host_to_device() override
  {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_device_ptr), m_host_ptr,
                          m_buffer_size * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copy_from_device_to_host() override
  {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_host_ptr), m_device_ptr,
                          m_buffer_size * sizeof(T), cudaMemcpyDeviceToHost));
  }

 private:
  uint32_t m_buffer_size = 0;

  T* m_host_ptr = nullptr;
  T* m_device_ptr = nullptr;
};

}  // namespace fredholm