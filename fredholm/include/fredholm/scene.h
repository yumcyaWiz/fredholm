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
#include "spdlog/spdlog.h"
#include "stb_image.h"
#include "sutil/vec_math.h"
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"
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

  Texture() {}
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

struct FloatTexture {
  uint32_t m_width;
  uint32_t m_height;
  std::vector<float4> m_data;

  FloatTexture(const std::filesystem::path& filepath)
  {
    spdlog::info("[Texture] loading {}", filepath.generic_string());

    int w, h, c;
    stbi_set_flip_vertically_on_load(false);
    float* img = stbi_loadf(filepath.c_str(), &w, &h, &c, STBI_rgb_alpha);
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

  void load_model(const std::filesystem::path& filepath)
  {
    clear();

    spdlog::info("[Scene] loading {}", filepath.generic_string());

    if (filepath.extension() == ".obj") {
      load_obj(filepath);
    } else if (filepath.extension() == ".gltf") {
      load_gltf(filepath);
    } else {
      throw std::runtime_error("failed to load " + filepath.generic_string() +
                               "\n" + "reason: invalid extension");
    }
  }

  void load_obj(const std::filesystem::path& filepath)
  {
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

    const auto parse_float = [](const std::string& str) {
      return std::stof(str);
    };
    const auto parse_float3 = [](const std::string& str) {
      // split string by space
      std::vector<std::string> tokens;
      std::stringstream ss(str);
      std::string buf;
      while (std::getline(ss, buf, ' ')) {
        if (!buf.empty()) { tokens.emplace_back(buf); }
      }

      if (tokens.size() != 3) {
        spdlog::error("invalid vec3 string");
        std::exit(EXIT_FAILURE);
      }

      // string to float conversion
      return make_float3(std::stof(tokens[0]), std::stof(tokens[1]),
                         std::stof(tokens[2]));
    };

    m_materials.resize(tinyobj_materials.size());
    for (int i = 0; i < m_materials.size(); ++i) {
      const auto& m = tinyobj_materials[i];

      // diffuse
      if (m.unknown_parameter.count("diffuse")) {
        m_materials[i].diffuse = parse_float(m.unknown_parameter.at("diffuse"));
      }

      // diffuse roughness
      if (m.unknown_parameter.count("diffuse_roughness")) {
        m_materials[i].diffuse_roughness =
            parse_float(m.unknown_parameter.at("diffuse_roughness"));
      }

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

      // sheen
      if (m.unknown_parameter.count("sheen")) {
        m_materials[i].sheen = parse_float(m.unknown_parameter.at("sheen"));
      }

      // sheen color
      if (m.unknown_parameter.count("sheen_color")) {
        m_materials[i].sheen_color =
            parse_float3(m.unknown_parameter.at("sheen_color"));
      }

      // sheen roughness
      if (m.unknown_parameter.count("sheen_roughness")) {
        m_materials[i].sheen_roughness =
            parse_float(m.unknown_parameter.at("sheen_roughness"));
      }

      // subsurface
      if (m.unknown_parameter.count("subsurface")) {
        m_materials[i].subsurface =
            parse_float(m.unknown_parameter.at("subsurface"));
      }

      // subsurface color
      if (m.unknown_parameter.count("subsurface_color")) {
        m_materials[i].subsurface_color =
            parse_float3(m.unknown_parameter.at("subsurface_color"));
      }

      // thin walled
      if (m.unknown_parameter.count("thin_walled")) {
        m_materials[i].thin_walled =
            parse_float(m.unknown_parameter.at("thin_walled"));
      }

      // emission
      if (m.emission[0] > 0 || m.emission[1] > 0 || m.emission[2] > 0) {
        m_materials[i].emission = 1.0f;
        m_materials[i].emission_color =
            make_float3(m.emission[0], m.emission[1], m.emission[2]);
      }

      // height map texture
      if (!m.bump_texname.empty()) {
        load_texture(filepath.parent_path(), m.bump_texname, false);
        m_materials[i].heightmap_texture_id = unique_textures[m.bump_texname];
      }

      // normal map texture
      if (!m.normal_texname.empty()) {
        load_texture(filepath.parent_path(), m.normal_texname, false);
        m_materials[i].normalmap_texture_id = unique_textures[m.normal_texname];
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

  void load_gltf(const std::filesystem::path& filepath)
  {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    const bool ret = loader.LoadASCIIFromFile(&model, &err, &warn,
                                              filepath.generic_string());

    if (!warn.empty()) { spdlog::warn("[tinygltf] {}", warn); }

    if (!err.empty()) {
      spdlog::error("[tinygltf] {}", err);
      throw std::runtime_error("failed to load " + filepath.generic_string());
    }

    if (!ret) {
      throw std::runtime_error("failed to load " + filepath.generic_string());
    }

    spdlog::info("[tinygltf] number of accessors: {}", model.accessors.size());
    spdlog::info("[tinygltf] number of animations: {}",
                 model.animations.size());
    spdlog::info("[tinygltf] number of buffers: {}", model.buffers.size());
    spdlog::info("[tinygltf] number of bufferViews: {}",
                 model.bufferViews.size());
    spdlog::info("[tinygltf] number of materials: {}", model.materials.size());
    spdlog::info("[tinygltf] number of meshes: {}", model.meshes.size());
    spdlog::info("[tinygltf] number of nodes: {}", model.nodes.size());
    spdlog::info("[tinygltf] number of textures: {}", model.textures.size());
    spdlog::info("[tinygltf] number of images: {}", model.images.size());
    spdlog::info("[tinygltf] number of skins: {}", model.skins.size());
    spdlog::info("[tinygltf] number of samplers: {}", model.samplers.size());
    spdlog::info("[tinygltf] number of cameras: {}", model.cameras.size());
    spdlog::info("[tinygltf] number of scenes: {}", model.scenes.size());
    spdlog::info("[tinygltf] number of lights: {}", model.lights.size());

    // load materials
    m_materials.resize(model.materials.size());
    for (int i = 0; i < model.materials.size(); ++i) {
      const auto& material = model.materials[i];
      spdlog::info("[tinygltf] loading material: {}", material.name);

      const auto& pmr = material.pbrMetallicRoughness;

      // base color
      m_materials[i].base_color =
          make_float3(pmr.baseColorFactor[0], pmr.baseColorFactor[1],
                      pmr.baseColorFactor[2]);

      // base color(texture)
      if (pmr.baseColorTexture.index != -1) {
        m_materials[i].base_color_texture_id = pmr.baseColorTexture.index;
      }
    }

    // load textures
    m_textures.resize(model.textures.size());
    for (int i = 0; i < model.textures.size(); ++i) {
      const auto& texture = model.textures[i];
      spdlog::info("[tinygltf] loading texture: {}", texture.name);

      const auto& image = model.images[texture.source];
      // TODO: set sRGB to Linear flag, create Texture Type instead?
      m_textures[i] = Texture(filepath.parent_path() / image.uri, true);
    }

    const auto get_buffer = [](const tinygltf::Model& model, int accessor_id,
                               int& stride, int& count) {
      const auto& accessor = model.accessors[accessor_id];
      const auto& bufferview = model.bufferViews[accessor.bufferView];
      const auto& buffer = model.buffers[bufferview.buffer];
      stride = accessor.ByteStride(bufferview);
      count = accessor.count;
      return buffer.data.data() + bufferview.byteOffset + accessor.byteOffset;
    };

    // load sub meshes
    int indices_offset = 0;
    int prev_indices_size = 0;
    for (const auto& mesh : model.meshes) {
      spdlog::info("[tinygltf] loading mesh: {}", mesh.name);
      spdlog::info("[tinygltf] number of primitives: {}",
                   mesh.primitives.size());

      // load primitives
      // NOTE: assuming each primitive has position, normal, texcoord
      for (const auto& primitive : mesh.primitives) {
        // indices
        int indices_stride, indices_count;
        const auto indices_raw =
            get_buffer(model, primitive.indices, indices_stride, indices_count);
        if (indices_stride != 2) {
          throw std::runtime_error("indices stride is not ushort");
        }

        const auto indices =
            reinterpret_cast<const unsigned short*>(indices_raw);
        for (int i = 0; i < indices_count / 3; ++i) {
          m_indices.push_back(make_uint3(indices[3 * i + 0] + indices_offset,
                                         indices[3 * i + 1] + indices_offset,
                                         indices[3 * i + 2] + indices_offset));
        }

        // positions, normals, texcoords
        int n_vertices = 0;
        for (const auto& attribute : primitive.attributes) {
          if (attribute.first == "POSITION") {
            int positions_stride, positions_count;
            const auto positions_raw = get_buffer(
                model, attribute.second, positions_stride, positions_count);
            if (positions_stride != 12) {
              throw std::runtime_error("positions stride is not float3");
            }

            const auto positions =
                reinterpret_cast<const float*>(positions_raw);

            for (int i = 0; i < positions_count; ++i) {
              m_vertices.push_back(make_float3(positions[3 * i + 0],
                                               positions[3 * i + 1],
                                               positions[3 * i + 2]));
            }

            n_vertices += positions_count;
          } else if (attribute.first == "NORMAL") {
            int normals_stride, normals_count;
            const auto normals_raw = get_buffer(model, attribute.second,
                                                normals_stride, normals_count);
            if (normals_stride != 12) {
              throw std::runtime_error("normals stride is not float3");
            }

            const auto normals = reinterpret_cast<const float*>(normals_raw);
            for (int i = 0; i < normals_count; ++i) {
              m_normals.push_back(make_float3(
                  normals[3 * i + 0], normals[3 * i + 1], normals[3 * i + 2]));
            }
          } else if (attribute.first == "TEXCOORD_0") {
            int texcoord_stride, texcoord_count;
            const auto texcoord_raw = get_buffer(
                model, attribute.second, texcoord_stride, texcoord_count);
            if (texcoord_stride != 8) {
              throw std::runtime_error("texcoord stride is not float2");
            }

            const auto texcoord = reinterpret_cast<const float*>(texcoord_raw);
            for (int i = 0; i < texcoord_count; ++i) {
              m_texcoords.push_back(
                  make_float2(texcoord[2 * i + 0], texcoord[2 * i + 1]));
            }
          }
        }

        // submesh offset, submesh nfaces
        m_submesh_offsets.push_back(prev_indices_size);
        m_submesh_n_faces.push_back(m_indices.size() - prev_indices_size);
        prev_indices_size = m_indices.size();

        // material id
        for (int i = 0; i < indices_count / 3; ++i) {
          m_material_ids.push_back(primitive.material);
        }

        indices_offset += n_vertices;
      }
    }

    spdlog::info("[tinygltf] number of sub meshes: {}",
                 m_submesh_offsets.size());
  }
};

}  // namespace fredholm