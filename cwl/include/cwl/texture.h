#pragma once
#include <cuda_runtime.h>

#include <filesystem>

#include "cwl/util.h"

namespace cwl
{

// RAII wrapper for CUDA texture object
template <typename T>
class CUDATexture
{
 public:
  CUDATexture(uint32_t width, uint32_t height, const T* data,
              bool srgb_to_linear = false)
  {
    cudaChannelFormatDesc channel_desc;
    channel_desc = cudaCreateChannelDesc<T>();

    // create array
    CUDA_CHECK(cudaMallocArray(&m_array, &channel_desc, width, height));

    // copy image data to array
    const uint32_t pitch = width * sizeof(T);
    CUDA_CHECK(cudaMemcpy2DToArray(m_array, 0, 0, data, pitch, pitch, height,
                                   cudaMemcpyHostToDevice));

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = m_array;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = std::is_same<T, uchar4>::value
                            ? cudaReadModeNormalizedFloat
                            : cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.sRGB = srgb_to_linear ? 1 : 0;

    // create texture object
    CUDA_CHECK(cudaCreateTextureObject(&m_texture_object, &res_desc, &tex_desc,
                                       nullptr));
  }

  CUDATexture(const CUDATexture& other) = delete;

  CUDATexture(CUDATexture&& other)
      : m_array(other.m_array), m_texture_object(other.m_texture_object)
  {
  }

  ~CUDATexture() noexcept(false)
  {
    CUDA_CHECK(cudaDestroyTextureObject(m_texture_object));
    CUDA_CHECK(cudaFreeArray(m_array));
  }

  cudaTextureObject_t get_texture_object() const { return m_texture_object; }

 private:
  cudaArray_t m_array = {};
  cudaTextureObject_t m_texture_object = {};
};

}  // namespace cwl