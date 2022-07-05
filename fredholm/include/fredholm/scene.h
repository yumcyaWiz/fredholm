#pragma once
#include <unistd.h>

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
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "sutil/vec_math.h"
//
#include "fredholm/shared.h"

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

struct Texture {
  uint32_t m_width;
  uint32_t m_height;
  std::vector<uchar4> m_data;
  bool m_srgb_to_linear;

  Texture(const std::filesystem::path& filepath, bool srgb_to_linear)
      : m_srgb_to_linear(srgb_to_linear)
  {
    spdlog::info("[Texture] loading {}", filepath.generic_string());

    // read image with stb_image
    int w, h, c;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* img =
        stbi_load(filepath.c_str(), &w, &h, &c, STBI_rgb_alpha);
    if (!img) {
      throw std::runtime_error("failed to load " + filepath.generic_string());
    }

    m_width = w;
    m_height = h;

    m_data.resize(m_width * m_height);
    for (int j = 0; j < m_height; ++j) {
      for (int i = 0; i < m_width; ++i) {
        const int idx_data = i + m_width * j;
        const int idx_img = 4 * i + 4 * m_width * j;
        m_data[idx_data].x = img[idx_img + 0];
        m_data[idx_data].y = img[idx_img + 1];
        m_data[idx_data].z = img[idx_img + 2];
        m_data[idx_data].w = img[idx_img + 3];
      }
    }

    stbi_image_free(img);
  }
};

// TODO: add transform in each submesh
struct Scene {
  // offset of each sub-mesh in index buffer
  std::vector<uint> m_submesh_offsets = {};
  // number of faces in each sub-mesh
  std::vector<uint> m_submesh_n_faces = {};

  std::vector<float3> m_vertices = {};
  std::vector<uint3> m_indices = {};
  std::vector<float2> m_texcoords = {};
  std::vector<float3> m_normals = {};
  std::vector<float3> m_tangents = {};
  // per-face material id
  std::vector<uint> m_material_ids = {};

  std::vector<Material> m_materials;
  std::vector<Texture> m_textures;

  Scene() {}

  bool is_valid() const
  {
    return m_submesh_offsets.size() > 0 && m_vertices.size() > 0 &&
           m_indices.size() > 0;
  }

  void clear()
  {
    m_submesh_offsets.clear();
    m_submesh_n_faces.clear();

    m_vertices.clear();
    m_indices.clear();
    m_texcoords.clear();
    m_normals.clear();
    m_tangents.clear();
    m_material_ids.clear();

    m_materials.clear();
    m_textures.clear();
  }

  void load_obj(const std::filesystem::path& filepath)
  {
    spdlog::info("loading {}", filepath.generic_string());

    clear();

    tinyobj::ObjReaderConfig reader_config;
    reader_config.triangulate = true;

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

    // load material and textures
    // key: texture filepath, value: texture id in m_textures
    std::unordered_map<std::string, unsigned int> unique_textures = {};

    const auto load_texture = [&](const std::filesystem::path& parent_filepath,
                                  const std::filesystem::path& filepath,
                                  bool srgb_to_linear) {
      if (unique_textures.count(filepath) == 0) {
        // load texture id
        unique_textures[filepath] = m_textures.size();
        // load texture
        m_textures.push_back(
            Texture(parent_filepath / filepath, srgb_to_linear));
      }
    };

    m_materials.resize(tinyobj_materials.size());
    for (int i = 0; i < m_materials.size(); ++i) {
      const auto& m = tinyobj_materials[i];

      // base color
      m_materials[i].base_color =
          make_float3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);

      // base color(texture)
      if (!m.diffuse_texname.empty()) {
        load_texture(filepath.parent_path(), m.diffuse_texname, true);

        // set texture id on material
        m_materials[i].base_color_texture_id =
            unique_textures[m.diffuse_texname];
      }

      // specular color
      m_materials[i].specular_color =
          make_float3(m.specular[0], m.specular[1], m.specular[2]);

      // specular color(texture)
      if (!m.specular_texname.empty()) {
        load_texture(filepath.parent_path(), m.specular_texname, false);
        m_materials[i].specular_color_texture_id =
            unique_textures[m.specular_texname];
      }

      // specular roughness
      if (m.roughness > 0) { m_materials[i].specular_roughness = m.roughness; }

      // specular roughness(texture)
      if (!m.roughness_texname.empty()) {
        load_texture(filepath.parent_path(), m.roughness_texname, false);
        m_materials[i].specular_roughness_texture_id =
            unique_textures[m.roughness_texname];
      }

      // metalness
      m_materials[i].metalness = m.metallic;

      // metalness texture
      if (!m.metallic_texname.empty()) {
        load_texture(filepath.parent_path(), m.metallic_texname, false);
        m_materials[i].metalness_texture_id =
            unique_textures[m.metallic_texname];
      }

      // coat
      if (m.clearcoat_thickness > 0) {
        m_materials[i].coat = m.clearcoat_thickness;
      }

      // coat roughness
      if (m.clearcoat_roughness > 0) {
        m_materials[i].coat_roughness = m.clearcoat_thickness;
      }

      // transmission
      m_materials[i].transmission = std::max(1.0f - m.dissolve, 0.0f);

      // transmission color
      if (m.transmittance[0] > 0 || m.transmittance[1] > 0 ||
          m.transmittance[2] > 0) {
        m_materials[i].transmission_color = make_float3(
            m.transmittance[0], m.transmittance[1], m.transmittance[2]);
      }

      // emission
      if (m.emission[0] > 0 || m.emission[1] > 0 || m.emission[2] > 0) {
        m_materials[i].emission = 1.0f;
        m_materials[i].emission_color =
            make_float3(m.emission[0], m.emission[1], m.emission[2]);
      }

      // alpha texture
      if (!m.alpha_texname.empty()) {
        load_texture(filepath.parent_path(), m.alpha_texname, false);
        m_materials[i].alpha_texture_id = unique_textures[m.alpha_texname];
      }
    }

    std::vector<Vertex> vertices{};
    std::unordered_map<Vertex, uint32_t> unique_vertices{};

    for (size_t s = 0; s < shapes.size(); ++s) {
      size_t index_offset = 0;

      std::vector<uint32_t> indices{};

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

        // load submesh per-face material id
        const int material_id = shapes[s].mesh.material_ids[f];
        m_material_ids.push_back(material_id);

        index_offset += 3;
      }

      spdlog::info("[tinyobjloader] submesh: {}, number of vertices: {}", s,
                   indices.size());
      spdlog::info("[tinyobjloader] submesh: {}, number of faces: {}", s,
                   indices.size() / 3);

      // fill submesh offset
      const size_t prev_indices_size = m_indices.size();
      m_submesh_offsets.push_back(prev_indices_size);

      // fill submesh indices
      for (int f = 0; f < indices.size() / 3; ++f) {
        m_indices.push_back(make_uint3(indices[3 * f + 0], indices[3 * f + 1],
                                       indices[3 * f + 2]));
      }

      // fill n_faces of sub-mesh
      m_submesh_n_faces.push_back(m_indices.size() - prev_indices_size);
    }

    // fill submesh vertices, normals, texcoords
    for (const auto& vertex : vertices) {
      m_vertices.push_back(
          make_float3(vertex.position.x, vertex.position.y, vertex.position.z));
      m_normals.push_back(
          make_float3(vertex.normal.x, vertex.normal.y, vertex.normal.z));
      m_texcoords.push_back(make_float2(vertex.texcoord.x, vertex.texcoord.y));
    }

    spdlog::info("[tinyobjloader] number of sub meshes: {}",
                 m_submesh_offsets.size());
    spdlog::info("[tinyobjloader] number of materials: {}", m_materials.size());
    spdlog::info("[tinyobjloader] number of textures: {}", m_textures.size());
  }
};

}  // namespace fredholm