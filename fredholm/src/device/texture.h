#pragma once
#include <cstdint>

#include "device/buffer.h"
#include "device/types.h"
#include "device/util.h"

namespace fredholm
{

template <typename T>
class Texture2D : public DeviceAndHostObject<T>
{
 public:
  Texture2D(uint32_t width, uint32_t height)
      : m_width(width), m_height(height), m_buffer(width * height)
  {
  }

  uint32_t get_width() const { return m_width; }

  uint32_t get_height() const { return m_height; }

  T get_value(uint32_t i, uint32_t j) const
  {
    return m_buffer.get_value(i + m_width * j);
  }

  void copy_from_host_to_device() override
  {
    m_buffer.copy_from_host_to_device();
  }
  void copy_from_device_to_host() override
  {
    m_buffer.copy_from_device_to_host();
  }

  uint32_t get_size_in_bytes() const override
  {
    return m_buffer.get_size_in_bytes();
  }

  T* get_host_ptr() const override { return m_buffer.get_host_ptr(); }

  T* get_device_ptr() const override { return m_buffer.get_device_ptr(); }

 private:
  uint32_t m_width = 0;
  uint32_t m_height = 0;

  Buffer<T> m_buffer;
};

}  // namespace fredholm