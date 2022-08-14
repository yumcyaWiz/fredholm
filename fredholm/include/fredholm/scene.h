#pragma once
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <set>
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

enum class TextureType { COLOR, NONCOLOR };

struct Texture {
  uint32_t m_width;
  uint32_t m_height;
  std::vector<uchar4> m_data;
  TextureType m_texture_type;

  Texture();
  Texture(const std::filesystem::path& filepath,
          const TextureType& texture_type);
};

struct FloatTexture {
  uint32_t m_width;
  uint32_t m_height;
  std::vector<float4> m_data;

  FloatTexture(const std::filesystem::path& filepath);
};

struct Node {
  int idx;  // tinygltf node index
  std::vector<Node> children;
  glm::mat4 transform;
  int camera_id;
  int submesh_id;
};

struct Animation {
  Node* node;  // target node

  std::vector<float> translation_input;       // key frame time
  std::vector<glm::vec3> translation_output;  // key frame translation

  std::vector<float> rotation_input;       // key frame time
  std::vector<glm::quat> rotation_output;  // key frame rotation

  std::vector<float> scale_input;       // key frame time
  std::vector<glm::vec3> scale_output;  // key frame scale
};

// TODO: add transform in each submesh
struct Scene {
  bool m_has_camera_transform = false;
  glm::mat4 m_camera_transform = {};

  // vertex data
  std::vector<float3> m_vertices = {};
  std::vector<uint3> m_indices = {};
  std::vector<float2> m_texcoords = {};
  std::vector<float3> m_normals = {};
  std::vector<float3> m_tangents = {};

  // per-face material id
  std::vector<uint> m_material_ids = {};

  std::vector<Material> m_materials;

  std::vector<Texture> m_textures;

  // offset of each sub-mesh in index buffer
  std::vector<uint> m_submesh_offsets = {};
  // number of faces in each sub-mesh
  std::vector<uint> m_submesh_n_faces = {};

  // per-face instance id
  std::vector<uint> m_instance_ids = {};

  // per-instance transform
  std::vector<glm::mat4> m_transforms = {};

  std::vector<Node> m_nodes = {};  // root nodes

  std::vector<Animation> m_animations = {};

  Scene();

  bool is_valid() const;

  void clear();

  void load_model(const std::filesystem::path& filepath);

  void load_obj(const std::filesystem::path& filepath);

  void load_gltf(const std::filesystem::path& filepath);

  Node load_gltf_node(const tinygltf::Model& model, int node_idx,
                      int& indices_offset, int& prev_indices_size);

  void update_transform();
  void update_transform_node(const Node& node, glm::mat4& transform);

  void update_animation(float time);

  Node* find_node(int node_idx);
  Node* find_node_node(Node& node, int node_idx);

  static const unsigned char* get_gltf_buffer(const tinygltf::Model& model,
                                              int accessor_id, int& stride,
                                              int& count);

  template <typename T>
  static T animation_linear_interpolate(const std::vector<float>& input,
                                        const std::vector<T>& output,
                                        float time)
  {
    const float t = std::fmod(time, input[input.size() - 1]);
    const int idx1 =
        std::lower_bound(input.begin(), input.end(), t) - input.begin();
    const int idx0 = std::max(idx1 - 1, 0);

    // linear interpolation
    const float h = t - input[idx0];
    const T output0 = output[idx0];
    const T output1 = output[idx1];
    return glm::mix(output0, output1, h);
  }
};

}  // namespace fredholm