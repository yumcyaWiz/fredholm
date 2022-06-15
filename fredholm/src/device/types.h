#pragma once
#include <cstdint>

namespace fredholm
{

// interface for object which is both on host and device
template <typename T>
class DeviceAndHostObject
{
  // copy object from host to device
  virtual void copy_from_host_to_device() = 0;

  // copy object from host to device
  virtual void copy_from_device_to_host() = 0;

  // get size of object in bytes
  virtual uint32_t get_size_in_bytes() const = 0;

  // get pointer to object on host
  virtual T* get_host_ptr() const = 0;

  // get pointer to object on device
  virtual T* get_device_ptr() const = 0;
};

}  // namespace fredholm