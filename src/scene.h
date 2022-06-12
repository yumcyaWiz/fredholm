#pragma once
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "device/buffer.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

struct Scene {
 public:
  Scene() {}

  uint32_t get_vertices_size() const { return m_vertices->get_size(); }

  uint32_t get_indices_size() const { return m_indices->get_size(); }

  float3* get_vertices_device_ptr() const
  {
    if (!m_vertices) { throw std::runtime_error("vertex buffer is empty"); }
    return m_vertices->get_device_ptr();
  }

  uint3* get_indices_device_ptr() const
  {
    if (!m_indices) { throw std::runtime_error("index buffer is empty"); }
    return m_indices->get_device_ptr();
  }

  float3* get_normal_device_ptr() const
  {
    if (!m_normals) { throw std::runtime_error("normal buffer is empty"); }
    return m_normals->get_device_ptr();
  }

  float2* get_texcoord_device_ptr() const
  {
    if (!m_texcoords) { throw std::runtime_error("texcoord buffer is empty"); }
    return m_texcoords->get_device_ptr();
  }

  void load_obj(const std::filesystem::path& filepath)
  {
    tinyobj::ObjReaderConfig reader_config;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filepath.generic_string(), reader_config)) {
      if (!reader.Error().empty()) {
        std::cerr << "tinyobjloader: " << reader.Error();
      }
      throw std::runtime_error("failed to load " + filepath.generic_string());
    }

    if (!reader.Warning().empty()) {
      std::cout << "tinyobjloader: " << reader.Warning();
    }

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();
    const auto& materials = reader.GetMaterials();

    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<float3> normals;
    std::vector<float2> texcoords;

    // TODO: remove vertex duplication, use index buffer instead.
    for (size_t s = 0; s < shapes.size(); ++s) {
      size_t index_offset = 0;
      for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
        size_t fv = static_cast<size_t>(shapes[s].mesh.num_face_vertices[f]);

        for (size_t v = 0; v < fv; ++v) {
          tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

          // vertex position
          tinyobj::real_t vx =
              attrib.vertices[3 * size_t(idx.vertex_index) + 0];
          tinyobj::real_t vy =
              attrib.vertices[3 * size_t(idx.vertex_index) + 1];
          tinyobj::real_t vz =
              attrib.vertices[3 * size_t(idx.vertex_index) + 2];
          vertices.push_back(make_float3(vx, vy, vz));

          // vertex normal
          if (idx.normal_index >= 0) {
            tinyobj::real_t nx =
                attrib.normals[3 * size_t(idx.normal_index) + 0];
            tinyobj::real_t ny =
                attrib.normals[3 * size_t(idx.normal_index) + 1];
            tinyobj::real_t nz =
                attrib.normals[3 * size_t(idx.normal_index) + 2];
            normals.push_back(make_float3(nx, ny, nz));
          }

          // texture coordinate
          if (idx.texcoord_index >= 0) {
            tinyobj::real_t tx =
                attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
            tinyobj::real_t ty =
                attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
            texcoords.push_back(make_float2(tx, ty));
          }
        }

        // fill indices buffer
        indices.push_back(make_uint3(3 * indices.size(), 3 * indices.size() + 1,
                                     3 * indices.size() + 2));

        index_offset += fv;
      }
    }

    // allocate and copy buffer from host to device
    m_vertices = std::make_unique<DeviceBuffer<float3>>(vertices.size());
    m_vertices->copy_from_host_to_device(vertices);

    m_indices = std::make_unique<DeviceBuffer<uint3>>(indices.size());
    m_indices->copy_from_host_to_device(indices);

    if (normals.size() > 0) {
      m_normals = std::make_unique<DeviceBuffer<float3>>(normals.size());
      m_normals->copy_from_host_to_device(normals);
    }
    if (texcoords.size() > 0) {
      m_texcoords = std::make_unique<DeviceBuffer<float2>>(texcoords.size());
      m_texcoords->copy_from_host_to_device(texcoords);
    }
  }

 private:
  std::unique_ptr<DeviceBuffer<float3>> m_vertices = nullptr;
  std::unique_ptr<DeviceBuffer<uint3>> m_indices = nullptr;
  std::unique_ptr<DeviceBuffer<float2>> m_texcoords = nullptr;
  std::unique_ptr<DeviceBuffer<float3>> m_normals = nullptr;
  std::unique_ptr<DeviceBuffer<float3>> m_tangents = nullptr;
};