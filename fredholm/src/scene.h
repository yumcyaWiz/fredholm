#pragma once
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"
#include "glm/gtx/hash.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
//
#include "spdlog/spdlog.h"
#include "sutil/vec_math.h"
//
#include "shared.h"

namespace fredholm
{

// https://vulkan-tutorial.com/Loading_models
// vertex deduplication
struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 texcoord;

  bool operator==(const Vertex& other) const
  {
    return position == other.position && normal == other.normal &&
           texcoord == other.texcoord;
  }
};

}  // namespace fredholm

namespace std
{
template <>
struct hash<fredholm::Vertex> {
  size_t operator()(fredholm::Vertex const& vertex) const
  {
    return ((hash<glm::vec3>()(vertex.position) ^
             (hash<glm::vec3>()(vertex.normal) << 1)) >>
            1) ^
           (hash<glm::vec2>()(vertex.texcoord) << 1);
  }
};

}  // namespace std

namespace fredholm
{

struct Scene {
  std::vector<float3> m_vertices = {};
  std::vector<uint3> m_indices = {};
  std::vector<float2> m_texcoords = {};
  std::vector<float3> m_normals = {};
  std::vector<float3> m_tangents = {};

  std::vector<Material> m_materials;
  // per-face material id
  std::vector<uint> m_material_ids;

  Scene() {}

  uint32_t n_vertices() const { return m_vertices.size(); }
  uint32_t n_faces() const { return m_indices.size(); }

  bool is_valid() const
  {
    return m_vertices.size() > 0 && m_indices.size() > 0;
  }

  void clear()
  {
    m_vertices.clear();
    m_indices.clear();
    m_texcoords.clear();
    m_normals.clear();
    m_tangents.clear();

    m_materials.clear();
    m_material_ids.clear();
  }

  void load_obj(const std::filesystem::path& filepath)
  {
    clear();

    tinyobj::ObjReaderConfig reader_config;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filepath.generic_string(), reader_config)) {
      if (!reader.Error().empty()) {
        spdlog::error("[tinyobjloader] {}", reader.Error());
      }
      throw std::runtime_error("failed to load " + filepath.generic_string());
    }

    if (!reader.Warning().empty()) {
      spdlog::warn("[tinyobjloader] {}", reader.Warning());
    }

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();
    const auto& tinyobj_materials = reader.GetMaterials();

    // load material
    m_materials.resize(tinyobj_materials.size());
    // TODO: load more parameters
    for (int i = 0; i < m_materials.size(); ++i) {
      const auto& m = tinyobj_materials[i];
      // base color
      m_materials[i].base_color =
          make_float3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);

      // emission
      if (m.emission[0] > 0 || m.emission[1] > 0 || m.emission[2] > 0) {
        m_materials[i].emission = 1.0f;
        m_materials[i].emission_color =
            make_float3(m.emission[0], m.emission[1], m.emission[2]);
      }
    }

    std::vector<Vertex> vertices{};
    std::vector<uint32_t> indices{};
    std::unordered_map<Vertex, uint32_t> unique_vertices{};

    for (size_t s = 0; s < shapes.size(); ++s) {
      size_t index_offset = 0;
      for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
        size_t fv = static_cast<size_t>(shapes[s].mesh.num_face_vertices[f]);

        if (fv != 3) { throw std::runtime_error("not a triangle face"); }

        std::vector<glm::vec3> vertices_temp;
        std::vector<glm::vec3> normals_temp;
        std::vector<glm::vec2> texcoords_temp;

        // NOTE: assuming fv = 3
        for (size_t v = 0; v < 3; ++v) {
          tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

          // vertex position
          tinyobj::real_t vx =
              attrib.vertices[3 * size_t(idx.vertex_index) + 0];
          tinyobj::real_t vy =
              attrib.vertices[3 * size_t(idx.vertex_index) + 1];
          tinyobj::real_t vz =
              attrib.vertices[3 * size_t(idx.vertex_index) + 2];
          vertices_temp.push_back(glm::vec3(vx, vy, vz));

          // vertex normal
          if (idx.normal_index >= 0) {
            tinyobj::real_t nx =
                attrib.normals[3 * size_t(idx.normal_index) + 0];
            tinyobj::real_t ny =
                attrib.normals[3 * size_t(idx.normal_index) + 1];
            tinyobj::real_t nz =
                attrib.normals[3 * size_t(idx.normal_index) + 2];
            normals_temp.push_back(glm::vec3(nx, ny, nz));
          }

          // texture coordinate
          if (idx.texcoord_index >= 0) {
            tinyobj::real_t tx =
                attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
            tinyobj::real_t ty =
                attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
            texcoords_temp.push_back(glm::vec2(tx, ty));
          }
        }

        // if vertex normals is empty, add face normal
        if (normals_temp.size() == 0) {
          const glm::vec3 v1 =
              glm::normalize(vertices_temp[1] - vertices_temp[0]);
          const glm::vec3 v2 =
              glm::normalize(vertices_temp[2] - vertices_temp[0]);
          const glm::vec3 n = glm::normalize(glm::cross(v1, v2));
          normals_temp.push_back(n);
          normals_temp.push_back(n);
          normals_temp.push_back(n);
        }

        // if texcoords is empty, add barycentric coords
        if (texcoords_temp.size() == 0) {
          texcoords_temp.push_back(glm::vec2(0, 0));
          texcoords_temp.push_back(glm::vec2(1, 0));
          texcoords_temp.push_back(glm::vec2(0, 1));
        }

        // fill vertices and indices
        for (int v = 0; v < 3; ++v) {
          Vertex vertex{};
          vertex.position = vertices_temp[v];
          vertex.normal = normals_temp[v];
          vertex.texcoord = texcoords_temp[v];

          if (unique_vertices.count(vertex) == 0) {
            unique_vertices[vertex] = static_cast<uint32_t>(vertices.size());
            vertices.push_back(vertex);
          }

          indices.push_back(unique_vertices[vertex]);
        }

        // load per-face material id
        const int material_id = shapes[s].mesh.material_ids[f];
        m_material_ids.push_back(material_id);

        index_offset += 3;
      }
    }

    // fill vertices, normals, texcoords
    for (const auto& vertex : vertices) {
      m_vertices.push_back(
          make_float3(vertex.position.x, vertex.position.y, vertex.position.z));
      m_normals.push_back(
          make_float3(vertex.normal.x, vertex.normal.y, vertex.normal.z));
      m_texcoords.push_back(make_float2(vertex.texcoord.x, vertex.texcoord.y));
    }

    // fill indices
    for (int f = 0; f < indices.size() / 3; ++f) {
      m_indices.push_back(make_uint3(indices[3 * f + 0], indices[3 * f + 1],
                                     indices[3 * f + 2]));
    }

    spdlog::info("[tinyobjloader] number of vertices: {}", m_vertices.size());
    spdlog::info("[tinyobjloader] number of faces: {}", m_indices.size());
  }
};

}  // namespace fredholm