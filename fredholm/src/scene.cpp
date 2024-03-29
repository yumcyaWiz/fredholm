#include "fredholm/scene.h"

using namespace fredholm;

Texture::Texture() {}

Texture::Texture(const std::filesystem::path& filepath,
                 const TextureType& texture_type)
    : m_texture_type(texture_type)
{
  spdlog::info("[Texture] loading {}", filepath.generic_string());

  // read image with stb_image
  int w, h, c;
  stbi_set_flip_vertically_on_load(true);
  unsigned char* img = stbi_load(filepath.c_str(), &w, &h, &c, STBI_rgb_alpha);
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

FloatTexture::FloatTexture(const std::filesystem::path& filepath)
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

Scene::Scene() {}

bool Scene::is_valid() const
{
  return m_submesh_offsets.size() > 0 && m_vertices.size() > 0 &&
         m_indices.size() > 0;
}

void Scene::clear()
{
  m_has_camera_transform = false;
  m_camera_transform = {};

  m_vertices.clear();
  m_indices.clear();
  m_texcoords.clear();
  m_normals.clear();
  m_tangents.clear();
  m_material_ids.clear();

  m_materials.clear();
  m_textures.clear();

  m_submesh_offsets.clear();
  m_submesh_n_faces.clear();

  m_instance_ids.clear();

  m_transforms.clear();

  m_nodes.clear();

  m_animations.clear();
}

void Scene::load_model(const std::filesystem::path& filepath, bool do_clear)
{
  if (do_clear) { clear(); }

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

void Scene::load_obj(const std::filesystem::path& filepath)
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
                                const TextureType& texture_type) {
    if (unique_textures.count(filepath) == 0) {
      // load texture id
      unique_textures[filepath] = m_textures.size();
      // load texture
      m_textures.push_back(Texture(parent_filepath / filepath, texture_type));
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

  for (int i = 0; i < tinyobj_materials.size(); ++i) {
    const auto& m = tinyobj_materials[i];

    Material mat;

    // diffuse
    if (m.unknown_parameter.count("diffuse")) {
      mat.diffuse = parse_float(m.unknown_parameter.at("diffuse"));
    }

    // diffuse roughness
    if (m.unknown_parameter.count("diffuse_roughness")) {
      mat.diffuse_roughness =
          parse_float(m.unknown_parameter.at("diffuse_roughness"));
    }

    // base color
    mat.base_color = make_float3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);

    // base color(texture)
    if (!m.diffuse_texname.empty()) {
      load_texture(filepath.parent_path(), m.diffuse_texname,
                   TextureType::COLOR);

      // set texture id on material
      mat.base_color_texture_id = unique_textures[m.diffuse_texname];
    }

    // specular color
    mat.specular_color =
        make_float3(m.specular[0], m.specular[1], m.specular[2]);

    // specular color(texture)
    if (!m.specular_texname.empty()) {
      load_texture(filepath.parent_path(), m.specular_texname,
                   TextureType::COLOR);
      mat.specular_color_texture_id = unique_textures[m.specular_texname];
    }

    // specular roughness
    if (m.roughness > 0) { mat.specular_roughness = m.roughness; }

    // specular roughness(texture)
    if (!m.roughness_texname.empty()) {
      load_texture(filepath.parent_path(), m.roughness_texname,
                   TextureType::NONCOLOR);
      mat.specular_roughness_texture_id = unique_textures[m.roughness_texname];
    }

    // metalness
    mat.metalness = m.metallic;

    // metalness texture
    if (!m.metallic_texname.empty()) {
      load_texture(filepath.parent_path(), m.metallic_texname,
                   TextureType::NONCOLOR);
      mat.metalness_texture_id = unique_textures[m.metallic_texname];
    }

    // coat
    if (m.clearcoat_thickness > 0) { mat.coat = m.clearcoat_thickness; }

    // coat roughness
    if (m.clearcoat_roughness > 0) {
      mat.coat_roughness = m.clearcoat_thickness;
    }

    // transmission
    mat.transmission = std::max(1.0f - m.dissolve, 0.0f);

    // transmission color
    if (m.transmittance[0] > 0 || m.transmittance[1] > 0 ||
        m.transmittance[2] > 0) {
      mat.transmission_color = make_float3(
          m.transmittance[0], m.transmittance[1], m.transmittance[2]);
    }

    // sheen
    if (m.unknown_parameter.count("sheen")) {
      mat.sheen = parse_float(m.unknown_parameter.at("sheen"));
    }

    // sheen color
    if (m.unknown_parameter.count("sheen_color")) {
      mat.sheen_color = parse_float3(m.unknown_parameter.at("sheen_color"));
    }

    // sheen roughness
    if (m.unknown_parameter.count("sheen_roughness")) {
      mat.sheen_roughness =
          parse_float(m.unknown_parameter.at("sheen_roughness"));
    }

    // subsurface
    if (m.unknown_parameter.count("subsurface")) {
      mat.subsurface = parse_float(m.unknown_parameter.at("subsurface"));
    }

    // subsurface color
    if (m.unknown_parameter.count("subsurface_color")) {
      mat.subsurface_color =
          parse_float3(m.unknown_parameter.at("subsurface_color"));
    }

    // thin walled
    if (m.unknown_parameter.count("thin_walled")) {
      mat.thin_walled = parse_float(m.unknown_parameter.at("thin_walled"));
    }

    // emission
    if (m.emission[0] > 0 || m.emission[1] > 0 || m.emission[2] > 0) {
      mat.emission = 1.0f;
      mat.emission_color =
          make_float3(m.emission[0], m.emission[1], m.emission[2]);
    }

    // height map texture
    if (!m.bump_texname.empty()) {
      load_texture(filepath.parent_path(), m.bump_texname,
                   TextureType::NONCOLOR);
      mat.heightmap_texture_id = unique_textures[m.bump_texname];
    }

    // normal map texture
    if (!m.normal_texname.empty()) {
      load_texture(filepath.parent_path(), m.normal_texname,
                   TextureType::NONCOLOR);
      mat.normalmap_texture_id = unique_textures[m.normal_texname];
    }

    // alpha texture
    if (!m.alpha_texname.empty()) {
      load_texture(filepath.parent_path(), m.alpha_texname,
                   TextureType::NONCOLOR);
      mat.alpha_texture_id = unique_textures[m.alpha_texname];
    }

    m_materials.push_back(mat);
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
        tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
        tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
        tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
        vertices_temp.push_back(glm::vec3(vx, vy, vz));

        // vertex normal
        if (idx.normal_index >= 0) {
          tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
          tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
          tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
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

    // fill transforms
    m_transforms.push_back(glm::identity<glm::mat4>());

    // fill instance ids
    // NOTE: there is no instance when loading obj
    for (int f = 0; f < indices.size() / 3; ++f) {
      m_instance_ids.push_back(0);
    }
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

void Scene::load_gltf(const std::filesystem::path& filepath)
{
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;

  const bool ret =
      loader.LoadASCIIFromFile(&model, &err, &warn, filepath.generic_string());

  if (!warn.empty()) { spdlog::warn("[tinygltf] {}", warn); }

  if (!err.empty()) {
    spdlog::error("[tinygltf] {}", err);
    throw std::runtime_error("failed to load " + filepath.generic_string());
  }

  if (!ret) {
    throw std::runtime_error("failed to load " + filepath.generic_string());
  }

  spdlog::info("[tinygltf] number of accessors: {}", model.accessors.size());
  spdlog::info("[tinygltf] number of animations: {}", model.animations.size());
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
  if (model.materials.size() == 0) {
    // throw std::runtime_error("there is no material");
  }

  for (int i = 0; i < model.materials.size(); ++i) {
    const auto& material = model.materials[i];
    spdlog::info("[tinygltf] loading material: {}", material.name);

    const auto& pmr = material.pbrMetallicRoughness;

    // base color
    Material mat;
    mat.base_color = make_float3(pmr.baseColorFactor[0], pmr.baseColorFactor[1],
                                 pmr.baseColorFactor[2]);

    // base color(texture)
    if (pmr.baseColorTexture.index != -1) {
      mat.base_color_texture_id = pmr.baseColorTexture.index;
    }

    // specular roughness
    mat.specular_roughness = pmr.roughnessFactor;

    // metalness
    mat.metalness = pmr.metallicFactor;

    // metallic roughness(texture)
    if (pmr.metallicRoughnessTexture.index != -1) {
      mat.metallic_roughness_texture_id = pmr.metallicRoughnessTexture.index;
    }

    if (material.extensions.contains("KHR_materials_clearcoat")) {
      const auto p = material.extensions.at("KHR_materials_clearcoat");

      // coat
      if (p.Has("clearcoatFactor")) {
        mat.coat = p.Get("clearcoatFactor").GetNumberAsDouble();
      }
      // coat(texture)
      if (p.Has("clearcoatTexture")) {
        mat.coat_texture_id = p.Get("clearcoatTexture").GetNumberAsInt();
      }
      // coat roughness
      if (p.Has("clearcoatRoughnessFactor")) {
        mat.coat_roughness =
            p.Get("clearcoatRoughnessFactor").GetNumberAsDouble();
      }
      // coat roughness(texture)
      if (p.Has("clearcoatRoughnessTexture")) {
        mat.coat_roughness_texture_id =
            p.Get("clearcoatRoughnessTexture").GetNumberAsInt();
      }
    }

    // emission
    if (material.emissiveFactor.size() == 3) {
      mat.emission = 1.0f;
      mat.emission_color =
          make_float3(material.emissiveFactor[0], material.emissiveFactor[1],
                      material.emissiveFactor[2]);
    }

    // emission texture
    if (material.emissiveTexture.index != -1) {
      mat.emission_texture_id = material.emissiveTexture.index;
    }

    // normal texture
    if (material.normalTexture.index != -1) {
      mat.normalmap_texture_id = material.normalTexture.index;
    }

    m_materials.push_back(mat);
  }

  // load textures
  for (int i = 0; i < model.textures.size(); ++i) {
    const auto& texture = model.textures[i];
    spdlog::info("[tinygltf] loading texture: {}", texture.name);

    const auto& image = model.images[texture.source];
    // TODO: set sRGB to Linear flag, create Texture Type instead?
    m_textures.push_back(
        Texture(filepath.parent_path() / image.uri, TextureType::NONCOLOR));
  }

  // load nodes
  int indices_offset = 0;
  int prev_indices_size = 0;
  for (const auto& node_idx : model.scenes[0].nodes) {
    m_nodes.push_back(
        load_gltf_node(model, node_idx, indices_offset, prev_indices_size));
  }

  // load transforms
  m_transforms.resize(m_submesh_offsets.size());
  update_transform();

  // load animations
  for (int i = 0; i < model.animations.size(); ++i) {
    Animation anim;

    const auto& animation = model.animations[i];

    // find target node
    anim.node = find_node(animation.channels[0].target_node);
    if (anim.node == nullptr) {
      throw std::runtime_error("invalid target node");
    }

    for (const auto& channel : animation.channels) {
      const auto& sampler = animation.samplers[channel.sampler];

      // load input
      int input_stride, input_count;
      const auto input_raw =
          get_gltf_buffer(model, sampler.input, input_stride, input_count);
      if (input_stride != 4) {
        throw std::runtime_error("unsupported animation input");
      }

      const auto input = reinterpret_cast<const float*>(input_raw);
      for (int input_idx = 0; input_idx < input_count; ++input_idx) {
        if (channel.target_path == "translation") {
          anim.translation_input.push_back(input[input_idx]);
        } else if (channel.target_path == "rotation") {
          anim.rotation_input.push_back(input[input_idx]);
        } else if (channel.target_path == "scale") {
          anim.scale_input.push_back(input[input_idx]);
        }
      }

      // load output
      int output_stride, output_count;
      const auto output_raw =
          get_gltf_buffer(model, sampler.output, output_stride, output_count);
      const auto output = reinterpret_cast<const float*>(output_raw);

      if (input_count != output_count) {
        throw std::runtime_error(
            "animation input size is not equal to output size");
      }

      if (channel.target_path == "translation") {
        if (output_stride != 12) {
          throw std::runtime_error("invalid output stride");
        }

        for (int output_idx = 0; output_idx < output_count; output_idx++) {
          anim.translation_output.push_back(
              glm::vec3(output[3 * output_idx + 0], output[3 * output_idx + 1],
                        output[3 * output_idx + 2]));
        }
      } else if (channel.target_path == "rotation") {
        if (output_stride != 16) {
          throw std::runtime_error("invalid output stride");
        }

        for (int output_idx = 0; output_idx < output_count; output_idx++) {
          anim.rotation_output.push_back(glm::quat(
              output[4 * output_idx + 3], output[4 * output_idx + 0],
              output[4 * output_idx + 1], output[4 * output_idx + 2]));
        }
      } else if (channel.target_path == "scale") {
        if (output_stride != 12) {
          throw std::runtime_error("invalid output stride");
        }

        for (int output_idx = 0; output_idx < output_count; output_idx++) {
          anim.scale_output.push_back(glm::vec3(output[3 * output_idx + 0],
                                                output[3 * output_idx + 1],
                                                output[3 * output_idx + 2]));
        }
      }
    }

    m_animations.push_back(anim);
  }

  spdlog::info("[Scene] number of sub meshes: {}", m_submesh_offsets.size());
  spdlog::info("[Scene] number of transforms: {}", m_transforms.size());
  spdlog::info("[Scene] number of animations: {}", m_animations.size());
}

Node Scene::load_gltf_node(const tinygltf::Model& model, int node_idx,
                           int& indices_offset, int& prev_indices_size)
{
  const tinygltf::Node& node = model.nodes[node_idx];
  spdlog::info("[tinygltf] loading node: {}", node.name);

  Node n;
  n.idx = node_idx;

  // load transform
  glm::vec3 translation = glm::vec3(0, 0, 0);
  if (node.translation.size() == 3) {
    translation.x = node.translation[0];
    translation.y = node.translation[1];
    translation.z = node.translation[2];
  }

  glm::quat rotation = glm::quat(1, 0, 0, 0);
  if (node.rotation.size() == 4) {
    rotation.x = node.rotation[0];
    rotation.y = node.rotation[1];
    rotation.z = node.rotation[2];
    rotation.w = node.rotation[3];
  }

  glm::vec3 scale = glm::vec3(1, 1, 1);
  if (node.scale.size() == 3) {
    scale.x = node.scale[0];
    scale.y = node.scale[1];
    scale.z = node.scale[2];
  }

  // create transform matrix
  glm::mat4 transform = glm::identity<glm::mat4>();
  transform = glm::translate(transform, translation);
  transform *= glm::mat4_cast(rotation);
  transform = glm::scale(transform, scale);

  if (node.matrix.size() == 16) {
    transform[0][0] = node.matrix[0];
    transform[0][1] = node.matrix[1];
    transform[0][2] = node.matrix[2];
    transform[0][3] = node.matrix[3];

    transform[1][0] = node.matrix[4];
    transform[1][1] = node.matrix[5];
    transform[1][2] = node.matrix[6];
    transform[1][3] = node.matrix[7];

    transform[2][0] = node.matrix[8];
    transform[2][1] = node.matrix[9];
    transform[2][2] = node.matrix[10];
    transform[2][3] = node.matrix[11];

    transform[3][0] = node.matrix[12];
    transform[3][1] = node.matrix[13];
    transform[3][2] = node.matrix[14];
    transform[3][3] = node.matrix[15];
  }

  n.transform = transform;

  // load mesh
  if (node.mesh != -1) {
    const auto& mesh = model.meshes[node.mesh];
    spdlog::info("[tinygltf] loading mesh: {}", mesh.name);
    spdlog::info("[tinygltf] number of primitives: {}", mesh.primitives.size());

    n.submesh_id = m_submesh_offsets.size();

    // load primitives
    // NOTE: assuming each primitive has position, normal, texcoord
    for (const auto& primitive : mesh.primitives) {
      // indices
      int indices_stride, indices_count;
      const auto indices_raw = get_gltf_buffer(model, primitive.indices,
                                               indices_stride, indices_count);
      if (indices_stride != 2) {
        throw std::runtime_error("indices stride is not ushort");
      }

      const auto indices = reinterpret_cast<const unsigned short*>(indices_raw);
      // const auto indices = reinterpret_cast<const unsigned
      // int*>(indices_raw);
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
          const auto positions_raw = get_gltf_buffer(
              model, attribute.second, positions_stride, positions_count);
          if (positions_stride != 12) {
            throw std::runtime_error("positions stride is not float3");
          }

          const auto positions = reinterpret_cast<const float*>(positions_raw);

          for (int i = 0; i < positions_count; ++i) {
            m_vertices.push_back(make_float3(positions[3 * i + 0],
                                             positions[3 * i + 1],
                                             positions[3 * i + 2]));
          }

          n_vertices += positions_count;
        } else if (attribute.first == "NORMAL") {
          int normals_stride, normals_count;
          const auto normals_raw = get_gltf_buffer(
              model, attribute.second, normals_stride, normals_count);
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
          const auto texcoord_raw = get_gltf_buffer(
              model, attribute.second, texcoord_stride, texcoord_count);
          if (texcoord_stride != 8) {
            throw std::runtime_error("texcoord stride is not float2");
          }

          const auto texcoord = reinterpret_cast<const float*>(texcoord_raw);
          for (int i = 0; i < texcoord_count; ++i) {
            m_texcoords.push_back(
                make_float2(texcoord[2 * i + 0], 1.0f - texcoord[2 * i + 1]));
          }
        }
      }

      // material id
      for (int i = 0; i < indices_count / 3; ++i) {
        m_material_ids.push_back(primitive.material);
      }

      // instance id
      for (int i = 0; i < indices_count / 3; ++i) {
        m_instance_ids.push_back(m_submesh_offsets.size());
      }

      indices_offset += n_vertices;
    }

    // submesh offset, submesh nfaces
    m_submesh_offsets.push_back(prev_indices_size);
    m_submesh_n_faces.push_back(m_indices.size() - prev_indices_size);
    prev_indices_size = m_indices.size();
  } else {
    n.submesh_id = -1;
  }

  // load children
  for (const auto child_idx : node.children) {
    n.children.push_back(
        load_gltf_node(model, child_idx, indices_offset, prev_indices_size));
  }

  return n;
}

void Scene::update_transform()
{
  for (const auto& node : m_nodes) {
    glm::mat4 transform = glm::identity<glm::mat4>();
    update_transform_node(node, transform);
  }
}

void Scene::update_transform_node(const Node& node, glm::mat4& transform)
{
  glm::mat4 m = transform * node.transform;

  // update camera transform
  if (node.camera_id != -1) {
    m_has_camera_transform = true;
    m_camera_transform = m;
  }

  // update submesh transform
  if (node.submesh_id != -1) { m_transforms[node.submesh_id] = m; }

  for (const auto& child_node : node.children) {
    update_transform_node(child_node, m);
  }
}

void Scene::update_animation(float time)
{
  for (const auto& animation : m_animations) {
    // translation
    glm::vec3 translation = glm::vec3(0, 0, 0);
    if (animation.translation_input.size() > 0) {
      translation = animation_linear_interpolate(
          animation.translation_input, animation.translation_output, time);
    }

    // rotation
    glm::quat rotation = glm::quat(1, 0, 0, 0);
    if (animation.rotation_input.size() > 0) {
      rotation = animation_linear_interpolate(animation.rotation_input,
                                              animation.rotation_output, time);
    }

    // scale
    glm::vec3 scale = glm::vec3(1, 1, 1);
    if (animation.scale_input.size() > 0) {
      scale = animation_linear_interpolate(animation.scale_input,
                                           animation.scale_output, time);
    }

    // compute transform matrix
    glm::mat4 transform = glm::identity<glm::mat4>();
    transform = glm::translate(transform, translation);
    transform *= glm::mat4_cast(rotation);
    transform = glm::scale(transform, scale);

    // update node
    animation.node->transform = transform;
  }

  // TODO: update only affected nodes
  update_transform();
}

Node* Scene::find_node(int node_idx)
{
  for (auto& node : m_nodes) {
    Node* ret = find_node_node(node, node_idx);
    if (ret) { return ret; }
  }

  return nullptr;
}

Node* Scene::find_node_node(Node& node, int node_idx)
{
  if (node.idx == node_idx) { return &node; }

  for (auto& child_node : node.children) {
    find_node_node(child_node, node_idx);
  }

  return nullptr;
}

const unsigned char* Scene::get_gltf_buffer(const tinygltf::Model& model,
                                            int accessor_id, int& stride,
                                            int& count)

{
  const auto& accessor = model.accessors[accessor_id];
  const auto& bufferview = model.bufferViews[accessor.bufferView];
  const auto& buffer = model.buffers[bufferview.buffer];
  stride = accessor.ByteStride(bufferview);
  count = accessor.count;
  return buffer.data.data() + bufferview.byteOffset + accessor.byteOffset;
}