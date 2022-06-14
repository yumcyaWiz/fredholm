#pragma once
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "shared.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

struct Scene {
  std::vector<float3> m_vertices = {};
  std::vector<uint3> m_indices = {};
  std::vector<float2> m_texcoords = {};
  std::vector<float3> m_normals = {};
  std::vector<float3> m_tangents = {};

  // per-face material
  std::vector<Material> m_materials;
  std::vector<uint> m_material_ids;

  Scene() {}

  uint32_t n_vertices() const { return m_vertices.size(); }
  uint32_t n_faces() const { return m_vertices.size() / 3; }

  bool is_valid() const { return m_vertices.size() > 0; }

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
        std::cerr << "tinyobjloader: " << reader.Error();
      }
      throw std::runtime_error("failed to load " + filepath.generic_string());
    }

    if (!reader.Warning().empty()) {
      std::cout << "tinyobjloader: " << reader.Warning();
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
          m_vertices.push_back(make_float3(vx, vy, vz));

          // vertex normal
          if (idx.normal_index >= 0) {
            tinyobj::real_t nx =
                attrib.normals[3 * size_t(idx.normal_index) + 0];
            tinyobj::real_t ny =
                attrib.normals[3 * size_t(idx.normal_index) + 1];
            tinyobj::real_t nz =
                attrib.normals[3 * size_t(idx.normal_index) + 2];
            m_normals.push_back(make_float3(nx, ny, nz));
          }

          // texture coordinate
          if (idx.texcoord_index >= 0) {
            tinyobj::real_t tx =
                attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
            tinyobj::real_t ty =
                attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
            m_texcoords.push_back(make_float2(tx, ty));
          }
        }

        // fill indices buffer
        m_indices.push_back(make_uint3(3 * m_indices.size(),
                                       3 * m_indices.size() + 1,
                                       3 * m_indices.size() + 2));

        index_offset += fv;

        // load per-face material id
        const int material_id = shapes[s].mesh.material_ids[f];
        m_material_ids.push_back(material_id);
      }
    }
  }
};