#pragma once
#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <vector>

#include "glm/glm.hpp"
#include "glm/gtx/hash.hpp"
#include "nlohmann/json.hpp"
#include "scene.h"
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"

using json = nlohmann::json;

namespace fredholm
{

// https://vulkan-tutorial.com/Loading_models
// vertex deduplication
struct Vertex
{
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
struct hash<fredholm::Vertex>
{
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

class SceneLoader
{
   public:
    static void load(const std::filesystem::path& filepath,
                     SceneGraph& scene_graph)
    {
        if (!scene_graph.is_empty()) { scene_graph.clear(); }

        if (filepath.extension() == ".obj")
        {
            SceneNode* node = load_obj(filepath, scene_graph);
            scene_graph.add_root(node);
        }
        else if (filepath.extension() == ".gltf")
        {
            std::vector<SceneNode*> nodes = load_gltf(filepath, scene_graph);
            for (const auto& node : nodes) { scene_graph.add_root(node); }
        }
        else if (filepath.extension() == ".json")
        {
            load_json(filepath, scene_graph);
        }
        else
        {
            throw std::runtime_error(
                std::format("unsupported file format: {}",
                            filepath.extension().generic_string()));
        }
    }

    static void load_envmap(const std::filesystem::path& filepath,
                            SceneGraph& scene_graph)
    {
        scene_graph.set_envmap(Texture{filepath, ColorSpace::SRGB});
    }

   private:
    static void load_json(const std::filesystem::path& filepath,
                          SceneGraph& scene_graph)
    {
        std::ifstream ifs(filepath);
        if (!ifs.is_open())
        {
            throw std::runtime_error(std::format(
                "failed to open json file {}\n", filepath.generic_string()));
        }

        json data = json::parse(ifs);

        if (data.contains("scene"))
        {
            for (const auto& scene_entry : data["scene"])
            {
                std::vector<SceneNode*> nodes = {};
                if (scene_entry.contains("file"))
                {
                    const std::filesystem::path file =
                        scene_entry["file"].get<std::string>();
                    if (file.extension() == ".obj")
                    {
                        SceneNode* ret = load_obj(filepath.parent_path() / file,
                                                  scene_graph);
                        nodes.push_back(ret);
                    }
                    else if (file.extension() == ".gltf")
                    {
                        std::vector<SceneNode*> ret = load_gltf(
                            filepath.parent_path() / file, scene_graph);
                        for (const auto& node : ret) { nodes.push_back(node); }
                    }
                    else
                    {
                        throw std::runtime_error(
                            std::format("unsupported file format: {}",
                                        file.extension().generic_string()));
                    }
                }
                else { throw std::runtime_error("file path is not specified"); }

                if (scene_entry.contains("transform"))
                {
                    const auto& transform = scene_entry["transform"];

                    glm::vec3 translate = {0, 0, 0};
                    if (transform.contains("translate"))
                    {
                        const auto& t = transform["translate"];
                        translate = {t[0], t[1], t[2]};
                    }
                    glm::quat rotate = glm::quat(1, 0, 0, 0);
                    if (transform.contains("rotate"))
                    {
                        const auto& r = transform["rotate"];
                        rotate = {r[0], r[1], r[2], r[3]};
                    }

                    glm::vec3 scale = {1, 1, 1};
                    if (transform.contains("scale"))
                    {
                        const auto& s = transform["scale"];
                        scale = {s[0], s[1], s[2]};
                    }

                    const glm::mat4 tmat =
                        glm::translate(glm::identity<glm::mat4>(), translate) *
                        glm::mat4_cast(rotate) *
                        glm::scale(glm::mat4(1.0f), scale);
                    for (const auto& node : nodes)
                    {
                        node->set_transform(tmat);
                    }
                }

                if (scene_entry.contains("material"))
                {
                    // TODO: load materials
                }

                for (const auto& node : nodes) { scene_graph.add_root(node); }
            }
        }
    }

    static SceneNode* load_obj(const std::filesystem::path& filepath,
                               SceneGraph& scene_graph)
    {
        std::vector<float3> m_vertices = {};
        std::vector<uint3> m_indices = {};
        std::vector<float3> m_normals = {};
        std::vector<float2> m_texcoords = {};
        std::vector<uint> m_material_ids = {};

        tinyobj::ObjReaderConfig reader_config;
        reader_config.triangulate = true;

        tinyobj::ObjReader reader;
        if (!reader.ParseFromFile(filepath.generic_string(), reader_config))
        {
            if (!reader.Error().empty())
            {
                spdlog::error("tinyobjloader: {}", reader.Error());
            }
            throw std::runtime_error(std::format("failed to load obj file {}\n",
                                                 filepath.generic_string()));
        }

        if (!reader.Warning().empty())
        {
            spdlog::warn("tinyobjloader: {}", reader.Warning());
        }

        const auto& attrib = reader.GetAttrib();
        const auto& shapes = reader.GetShapes();
        const auto& tinyobj_materials = reader.GetMaterials();

        // load material and textures
        // key: texture filepath, value: texture id in m_textures
        std::unordered_map<std::string, unsigned int> unique_textures = {};

        const auto load_texture =
            [&](const std::filesystem::path& parent_filepath,
                const std::filesystem::path& filepath,
                const ColorSpace& color_space)
        {
            if (unique_textures.count(filepath) == 0)
            {
                // load texture id
                unique_textures[filepath] = scene_graph.n_textures();
                // load texture
                scene_graph.add_texture(
                    Texture(parent_filepath / filepath, color_space));
            }
        };

        const auto parse_float = [](const std::string& str)
        { return std::stof(str); };
        const auto parse_float3 = [](const std::string& str)
        {
            // split string by space
            std::vector<std::string> tokens;
            std::stringstream ss(str);
            std::string buf;
            while (std::getline(ss, buf, ' '))
            {
                if (!buf.empty()) { tokens.emplace_back(buf); }
            }

            if (tokens.size() != 3)
            {
                spdlog::error("invalid vec3 string");
                std::exit(EXIT_FAILURE);
            }

            // string to float conversion
            return make_float3(std::stof(tokens[0]), std::stof(tokens[1]),
                               std::stof(tokens[2]));
        };

        // load materials
        const uint32_t material_id_offset = scene_graph.n_materials();
        for (int i = 0; i < tinyobj_materials.size(); ++i)
        {
            const auto& m = tinyobj_materials[i];

            Material mat;

            // diffuse
            if (m.unknown_parameter.count("diffuse"))
            {
                mat.diffuse = parse_float(m.unknown_parameter.at("diffuse"));
            }

            // diffuse roughness
            if (m.unknown_parameter.count("diffuse_roughness"))
            {
                mat.diffuse_roughness =
                    parse_float(m.unknown_parameter.at("diffuse_roughness"));
            }

            // base color
            mat.base_color =
                make_float3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);

            // base color(texture)
            if (!m.diffuse_texname.empty())
            {
                load_texture(filepath.parent_path(), m.diffuse_texname,
                             ColorSpace::SRGB);

                mat.base_color_texture_id = unique_textures[m.diffuse_texname];
            }

            // specular color
            mat.specular_color =
                make_float3(m.specular[0], m.specular[1], m.specular[2]);

            // specular color(texture)
            if (!m.specular_texname.empty())
            {
                load_texture(filepath.parent_path(), m.specular_texname,
                             ColorSpace::LINEAR);

                mat.specular_color_texture_id =
                    unique_textures[m.specular_texname];
            }

            // specular roughness
            if (m.roughness > 0.0f) { mat.specular_roughness = m.roughness; }

            // specular roughness(texture)
            if (!m.roughness_texname.empty())
            {
                load_texture(filepath.parent_path(), m.roughness_texname,
                             ColorSpace::LINEAR);

                mat.specular_roughness_texture_id =
                    unique_textures[m.roughness_texname];
            }

            // metalness
            mat.metalness = m.metallic;

            // metalness(texture)
            if (!m.metallic_texname.empty())
            {
                load_texture(filepath.parent_path(), m.metallic_texname,
                             ColorSpace::LINEAR);

                mat.metalness_texture_id = unique_textures[m.metallic_texname];
            }

            // coat
            if (m.clearcoat_thickness > 0.0f)
            {
                mat.coat = m.clearcoat_thickness;
            }

            // coat roughness
            if (m.clearcoat_roughness > 0.0f)
            {
                mat.coat_roughness = m.clearcoat_roughness;
            }

            // transmission
            mat.transmission = std::max(1.0f - m.dissolve, 0.0f);

            // transmission color
            if (m.transmittance[0] > 0.0f || m.transmittance[0] > 0.0f ||
                m.transmittance[2] > 0.0f)
            {
                mat.transmission_color = make_float3(
                    m.transmittance[0], m.transmittance[1], m.transmittance[2]);
            }

            // sheen
            if (m.unknown_parameter.count("sheen"))
            {
                mat.sheen = parse_float(m.unknown_parameter.at("sheen"));
            }

            // sheen color
            if (m.unknown_parameter.count("sheen_color"))
            {
                mat.sheen_color =
                    parse_float3(m.unknown_parameter.at("sheen_color"));
            }

            // sheen roughness
            if (m.unknown_parameter.count("sheen_roughness"))
            {
                mat.sheen_roughness =
                    parse_float(m.unknown_parameter.at("sheen_roughness"));
            }

            // subsurface
            if (m.unknown_parameter.count("subsurface"))
            {
                mat.subsurface =
                    parse_float(m.unknown_parameter.at("subsurface"));
            }

            // subsurface color
            if (m.unknown_parameter.count("subsurface_color"))
            {
                mat.subsurface_color =
                    parse_float3(m.unknown_parameter.at("subsurface_color"));
            }

            // thin walled
            if (m.unknown_parameter.count("thin_walled"))
            {
                mat.thin_walled =
                    parse_float(m.unknown_parameter.at("thin_walled"));
            }

            // emission
            if (m.emission[0] > 0 || m.emission[1] > 0 || m.emission[2] > 0)
            {
                mat.emission = 1.0f;
                mat.emission_color =
                    make_float3(m.emission[0], m.emission[1], m.emission[2]);
            }

            // height map texture
            if (!m.bump_texname.empty())
            {
                load_texture(filepath.parent_path(), m.bump_texname,
                             ColorSpace::LINEAR);

                mat.heightmap_texture_id = unique_textures[m.bump_texname];
            }

            // normal map texture
            if (!m.normal_texname.empty())
            {
                load_texture(filepath.parent_path(), m.normal_texname,
                             ColorSpace::LINEAR);

                mat.normalmap_texture_id = unique_textures[m.normal_texname];
            }

            // alpha texture
            if (!m.alpha_texname.empty())
            {
                load_texture(filepath.parent_path(), m.alpha_texname,
                             ColorSpace::LINEAR);

                mat.alpha_texture_id = unique_textures[m.alpha_texname];
            }

            scene_graph.add_material(mat);
        }

        std::vector<Vertex> unique_vertices = {};
        std::unordered_map<Vertex, uint32_t> unique_vertex_indices = {};
        std::vector<uint32_t> indices = {};

        for (size_t s = 0; s < shapes.size(); ++s)
        {
            size_t index_offset = 0;

            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f)
            {
                const size_t fv = shapes[s].mesh.num_face_vertices[f];
                if (fv != 3)
                {
                    throw std::runtime_error(
                        "non-triangle faces are not supported");
                }

                std::vector<glm::vec3> vertices_temp;
                std::vector<glm::vec3> normals_temp;
                std::vector<glm::vec2> texcoords_temp;

                for (size_t v = 0; v < 3; ++v)
                {
                    const tinyobj::index_t idx =
                        shapes[s].mesh.indices[index_offset + v];

                    // vertex position
                    const glm::vec3 vertex = {
                        attrib.vertices[3 * idx.vertex_index + 0],
                        attrib.vertices[3 * idx.vertex_index + 1],
                        attrib.vertices[3 * idx.vertex_index + 2]};
                    vertices_temp.push_back(vertex);

                    // vertex normal
                    if (idx.normal_index >= 0)
                    {
                        const glm::vec3 normal = {
                            attrib.normals[3 * idx.normal_index + 0],
                            attrib.normals[3 * idx.normal_index + 1],
                            attrib.normals[3 * idx.normal_index + 2]};
                        normals_temp.push_back(normal);
                    }

                    // vertex texcoord
                    if (idx.texcoord_index >= 0)
                    {
                        const glm::vec2 texcoord = {
                            attrib.texcoords[2 * idx.texcoord_index + 0],
                            attrib.texcoords[2 * idx.texcoord_index + 1]};
                        texcoords_temp.push_back(texcoord);
                    }
                }

                // if vertex normal is empty, use face normal instead
                if (normals_temp.size() == 0)
                {
                    const glm::vec3 normal = glm::normalize(
                        glm::cross(vertices_temp[1] - vertices_temp[0],
                                   vertices_temp[2] - vertices_temp[0]));
                    normals_temp.push_back(normal);
                    normals_temp.push_back(normal);
                    normals_temp.push_back(normal);
                }

                // if texcoord is empty, use barycentric coordinates instead
                if (texcoords_temp.size() == 0)
                {
                    texcoords_temp.push_back(glm::vec2(0, 0));
                    texcoords_temp.push_back(glm::vec2(1, 0));
                    texcoords_temp.push_back(glm::vec2(0, 1));
                }

                for (size_t v = 0; v < 3; ++v)
                {
                    Vertex vertex = {};
                    vertex.position = vertices_temp[v];
                    vertex.normal = normals_temp[v];
                    vertex.texcoord = texcoords_temp[v];

                    if (unique_vertex_indices.count(vertex) == 0)
                    {
                        unique_vertex_indices[vertex] =
                            static_cast<uint32_t>(unique_vertices.size());
                        unique_vertices.push_back(vertex);
                    }
                    indices.push_back(unique_vertex_indices[vertex]);
                }

                const int material_id =
                    shapes[s].mesh.material_ids[f] + material_id_offset;
                m_material_ids.push_back(material_id);

                index_offset += fv;
            }
        }

        for (const auto& vertex : unique_vertices)
        {
            m_vertices.push_back(make_float3(
                vertex.position.x, vertex.position.y, vertex.position.z));
            m_normals.push_back(
                make_float3(vertex.normal.x, vertex.normal.y, vertex.normal.z));
            m_texcoords.push_back(
                make_float2(vertex.texcoord.x, vertex.texcoord.y));
        }

        for (size_t i = 0; i < indices.size(); i += 3)
        {
            m_indices.push_back(
                make_uint3(indices[i], indices[i + 1], indices[i + 2]));
        }

        spdlog::info("loaded obj file {}", filepath.generic_string());
        spdlog::info("# of vertices: {}", m_vertices.size());
        spdlog::info("# of faces: {}", m_indices.size());
        spdlog::info("# of materials: {}", scene_graph.n_materials());
        spdlog::info("# of textures: {}", scene_graph.n_textures());

        GeometryNode* geometry = new GeometryNode(
            std::move(m_vertices), std::move(m_indices), std::move(m_normals),
            std::move(m_texcoords), std::move(m_material_ids));

        return geometry;
    }

    static std::vector<SceneNode*> load_gltf(
        const std::filesystem::path& filepath, SceneGraph& scene_graph)
    {
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        std::string err;
        std::string warn;

        const bool loaded = loader.LoadASCIIFromFile(&model, &err, &warn,
                                                     filepath.generic_string());
        if (!warn.empty()) { spdlog::warn("tinygltf: {}", warn); }
        if (!err.empty()) { spdlog::error("tinygltf: {}", err); }
        if (!loaded)
        {
            throw std::runtime_error(std::format(
                "failed to load gltf file {}\n", filepath.generic_string()));
        }

        spdlog::info("number of nodes: {}", model.nodes.size());
        spdlog::info("number of buffers: {}", model.buffers.size());
        spdlog::info("number of buffer views: {}", model.bufferViews.size());
        spdlog::info("number of meshes: {}", model.meshes.size());
        spdlog::info("number of accessors: {}", model.accessors.size());
        spdlog::info("number of images: {}", model.images.size());
        spdlog::info("number of samplers: {}", model.samplers.size());

        spdlog::info("number of scenes: {}", model.scenes.size());
        spdlog::info("number of cameras: {}", model.cameras.size());
        spdlog::info("number of lights: {}", model.lights.size());
        spdlog::info("number of animations: {}", model.animations.size());
        spdlog::info("number of materials: {}", model.materials.size());
        spdlog::info("number of textures: {}", model.textures.size());

        // load materials
        const uint32_t material_id_offset = scene_graph.n_materials();
        const uint32_t texture_id_offset = scene_graph.n_textures();
        for (int i = 0; i < model.materials.size(); ++i)
        {
            const auto& material = model.materials[i];
            const Material& m = load_gltf_material(material, texture_id_offset);
            scene_graph.add_material(m);
        }

        // load textures
        for (int i = 0; i < model.textures.size(); ++i)
        {
            const auto& texture = model.textures[i];
            const Texture t = load_gltf_texture(texture, model, filepath);
            scene_graph.add_texture(t);
        }

        // load nodes
        std::vector<SceneNode*> ret = {};
        for (const auto& node_idx : model.scenes[0].nodes)
        {
            SceneNode* root = new SceneNode();
            ret.push_back(root);

            const auto& node = model.nodes[node_idx];
            load_gltf_node(node, model, root, material_id_offset);
        }

        return ret;
    }

    static Texture load_gltf_texture(
        const tinygltf::Texture& texture, const tinygltf::Model& model,
        const std::filesystem::path& parent_filepath)
    {
        spdlog::info("loading texture: {}", texture.name);
        const auto& image = model.images[texture.source];
        // TODO: set colorspace correctly
        return Texture(parent_filepath.parent_path() / image.uri,
                       ColorSpace::SRGB);
    }

    static Material load_gltf_material(const tinygltf::Material& material,
                                       uint32_t texture_id_offset)
    {
        spdlog::info("loading material: {}", material.name);

        Material ret;

        const auto& pmr = material.pbrMetallicRoughness;

        // base color
        ret.base_color =
            make_float3(pmr.baseColorFactor[0], pmr.baseColorFactor[1],
                        pmr.baseColorFactor[2]);

        // base color(texture)
        if (pmr.baseColorTexture.index != -1)
        {
            ret.base_color_texture_id =
                pmr.baseColorTexture.index + texture_id_offset;
        }

        // specular roughness
        ret.specular_roughness = pmr.roughnessFactor;

        // metalness
        ret.metalness = pmr.metallicFactor;

        // metallic roughness(texture)
        if (pmr.metallicRoughnessTexture.index != -1)
        {
            ret.metalness_texture_id =
                pmr.metallicRoughnessTexture.index + texture_id_offset;
        }

        // clearcoat
        if (material.extensions.contains("KHR_materials_clearcoat"))
        {
            const auto& p = material.extensions.at("KHR_materials_clearcoat");

            // coat
            if (p.Has("clearcoatFactor"))
            {
                ret.coat = p.Get("clearcoatFactor").GetNumberAsDouble();
            }

            // coat(texture)
            if (p.Has("clearcoatTexture"))
            {
                ret.coat_texture_id =
                    p.Get("clearcoatTexture").GetNumberAsInt();
            }

            // coat roughness
            if (p.Has("clearcoatRoughnessFactor"))
            {
                ret.coat_roughness =
                    p.Get("clearcoatRoughnessFactor").GetNumberAsDouble();
            }

            // coat roughness(texture)
            if (p.Has("clearcoatRoughnessTexture"))
            {
                ret.coat_roughness_texture_id =
                    p.Get("clearcoatRoughnessTexture").GetNumberAsInt();
            }
        }

        // TODO: load KHR_materials_anisotropy
        // TODO: load KHR_materials_ior
        // TODO: load KHR_materials_iridescence
        // TODO: load KHR_materials_sheen
        // TODO: load KHR_materials_specular
        // TODO: load KHR_materials_transmission
        // TODO: load KHR_materials_volume

        if (material.extensions.contains("KHR_materials_specular"))
        {
            const auto& p = material.extensions.at("KHR_materials_specular");

            // specular
            if (p.Has("specularFactor"))
            {
                ret.specular = p.Get("specularFactor").GetNumberAsDouble();
            }
            // specular color
            if (p.Has("specularColorFactor"))
            {
                const auto& c = p.Get("specularColorFactor");
                ret.specular_color = make_float3(c.Get(0).GetNumberAsDouble(),
                                                 c.Get(1).GetNumberAsDouble(),
                                                 c.Get(2).GetNumberAsDouble());
            }
            // specular texture
            if (p.Has("specularTexture"))
            {
                ret.specular = 1.0f;
                ret.specular_color_texture_id =
                    p.Get("specularTexture").GetNumberAsInt();
            }
            // specular color texture
            if (p.Has("specularColorTexture"))
            {
                ret.specular = 1.0f;
                ret.specular_color_texture_id =
                    p.Get("specularColorTexture").GetNumberAsInt();
            }
        }

        // emission
        if (material.emissiveFactor.size() == 3)
        {
            ret.emission = 1.0f;
            ret.emission_color = make_float3(material.emissiveFactor[0],
                                             material.emissiveFactor[1],
                                             material.emissiveFactor[2]);
        }

        // emission texture
        if (material.emissiveTexture.index != -1)
        {
            ret.emission_texture_id =
                material.emissiveTexture.index + texture_id_offset;
        }

        // normal texture
        if (material.normalTexture.index != -1)
        {
            ret.normalmap_texture_id =
                material.normalTexture.index + texture_id_offset;
        }

        return ret;
    }

    static glm::mat4 load_gltf_transform(const tinygltf::Node& node)
    {
        glm::vec3 translation = {0, 0, 0};
        if (node.translation.size() == 3)
        {
            translation = {node.translation[0], node.translation[1],
                           node.translation[2]};
        }

        glm::quat rotation = glm::quat(1, 0, 0, 0);
        if (node.rotation.size() == 4)
        {
            rotation = {static_cast<float>(node.rotation[3]),
                        static_cast<float>(node.rotation[0]),
                        static_cast<float>(node.rotation[1]),
                        static_cast<float>(node.rotation[2])};
        }

        glm::vec3 scale = {1, 1, 1};
        if (node.scale.size() == 3)
        {
            scale = {node.scale[0], node.scale[1], node.scale[2]};
        }

        glm::mat4 transform =
            glm::translate(glm::identity<glm::mat4>(), translation) *
            glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);

        if (node.matrix.size() == 16)
        {
            transform = glm::make_mat4(node.matrix.data());
        }

        return transform;
    }

    static const unsigned char* get_gltf_buffer_data(
        const tinygltf::Model& model, int accessor_id, uint32_t& stride,
        uint32_t& count)
    {
        const auto& accessor = model.accessors[accessor_id];
        const auto& buffer_view = model.bufferViews[accessor.bufferView];
        const auto& buffer = model.buffers[buffer_view.buffer];

        stride = accessor.ByteStride(buffer_view);
        count = accessor.count;

        return buffer.data.data() + accessor.byteOffset +
               buffer_view.byteOffset;
    }

    static std::vector<GeometryNode*> load_gltf_mesh(
        const tinygltf::Mesh& mesh, const tinygltf::Model& model,
        uint32_t material_id_offset)
    {
        spdlog::info("loading mesh: {}", mesh.name);
        spdlog::info("number of primitives: {}", mesh.primitives.size());

        std::vector<GeometryNode*> ret = {};

        for (const auto& primitive : mesh.primitives)
        {
            std::vector<float3> vertices = {};
            std::vector<uint3> indices = {};
            std::vector<float3> normals = {};
            std::vector<float2> texcoords = {};
            std::vector<uint> material_ids = {};

            // indices
            {
                uint32_t indices_stride, indices_count = 0;
                const auto buffer_raw = get_gltf_buffer_data(
                    model, primitive.indices, indices_stride, indices_count);
                if (indices_stride == 2)
                {
                    const auto buffer =
                        reinterpret_cast<const unsigned short*>(buffer_raw);
                    for (int i = 0; i < indices_count / 3; ++i)
                    {
                        const uint3 idx = {buffer[3 * i + 0], buffer[3 * i + 1],
                                           buffer[3 * i + 2]};
                        indices.push_back(idx);
                    }
                }
                else if (indices_stride == 4)
                {
                    const auto buffer =
                        reinterpret_cast<const unsigned int*>(buffer_raw);
                    for (int i = 0; i < indices_count / 3; ++i)
                    {
                        const uint3 idx = {buffer[3 * i + 0], buffer[3 * i + 1],
                                           buffer[3 * i + 2]};
                        indices.push_back(idx);
                    }
                }
                else
                {
                    throw std::runtime_error(
                        "indices stride must be 2 or 4 bytes");
                }
            }

            // positions
            if (primitive.attributes.contains("POSITION"))
            {
                uint32_t positions_stride, positions_count = 0;
                const auto buffer_raw = get_gltf_buffer_data(
                    model, primitive.attributes.at("POSITION"),
                    positions_stride, positions_count);
                if (positions_stride != 12)
                {
                    throw std::runtime_error(
                        "positions stride must be 12 bytes");
                }
                const auto buffer = reinterpret_cast<const float*>(buffer_raw);
                for (int i = 0; i < positions_count; ++i)
                {
                    const float3 pos = {buffer[3 * i + 0], buffer[3 * i + 1],
                                        buffer[3 * i + 2]};
                    vertices.push_back(pos);
                }
            }
            else { throw std::runtime_error("POSITION attribute is required"); }

            // normals
            if (primitive.attributes.contains("NORMAL"))
            {
                uint32_t normals_stride, normals_count = 0;
                const auto buffer_raw = get_gltf_buffer_data(
                    model, primitive.attributes.at("NORMAL"), normals_stride,
                    normals_count);
                if (normals_stride != 12)
                {
                    throw std::runtime_error("normals stride must be 12 bytes");
                }
                const auto buffer = reinterpret_cast<const float*>(buffer_raw);
                for (int i = 0; i < normals_count; ++i)
                {
                    const float3 normal = {buffer[3 * i + 0], buffer[3 * i + 1],
                                           buffer[3 * i + 2]};
                    normals.push_back(normal);
                }
            }
            else
            {
                // use face normal instead
                uint32_t positions_stride, positions_count = 0;
                const auto buffer_raw = get_gltf_buffer_data(
                    model, primitive.attributes.at("POSITION"),
                    positions_stride, positions_count);
                if (positions_stride != 12)
                {
                    throw std::runtime_error(
                        "positions stride must be 12 bytes");
                }
                const auto buffer = reinterpret_cast<const float*>(buffer_raw);
                for (int i = 0; i < positions_count; i += 3)
                {
                    const glm::vec3 pos0 = {buffer[3 * i + 0],
                                            buffer[3 * i + 1],
                                            buffer[3 * i + 2]};
                    const glm::vec3 pos1 = {buffer[3 * (i + 1) + 0],
                                            buffer[3 * (i + 1) + 1],
                                            buffer[3 * (i + 1) + 2]};
                    const glm::vec3 pos2 = {buffer[3 * (i + 2) + 0],
                                            buffer[3 * (i + 2) + 1],
                                            buffer[3 * (i + 2) + 2]};
                    const glm::vec3 normal =
                        glm::normalize(glm::cross(pos1 - pos0, pos2 - pos0));
                    normals.push_back(
                        make_float3(normal.x, normal.y, normal.z));
                }
            }

            // texcoords
            if (primitive.attributes.contains("TEXCOORD_0"))
            {
                uint32_t texcoords_stride, texcoords_count = 0;
                const auto buffer_raw = get_gltf_buffer_data(
                    model, primitive.attributes.at("TEXCOORD_0"),
                    texcoords_stride, texcoords_count);
                if (texcoords_stride != 8)
                {
                    throw std::runtime_error(
                        "texcoords stride must be 8 bytes");
                }
                const auto buffer = reinterpret_cast<const float*>(buffer_raw);
                for (int i = 0; i < texcoords_count; ++i)
                {
                    const float2 texcoord = {buffer[2 * i + 0],
                                             buffer[2 * i + 1]};
                    texcoords.push_back(texcoord);
                }
            }
            else
            {
                // use barycentric coordinates instead
                for (int i = 0; i < indices.size(); ++i)
                {
                    texcoords.push_back(make_float2(0, 0));
                    texcoords.push_back(make_float2(1, 0));
                    texcoords.push_back(make_float2(0, 1));
                }
            }

            // material ids
            {
                uint32_t material_id = primitive.material + material_id_offset;
                // some scenes could have invalid material id
                if (primitive.material == -1)
                {
                    material_id = material_id_offset;
                    spdlog::warn("material id must be specified");
                }

                for (int i = 0; i < indices.size(); ++i)
                {
                    material_ids.push_back(material_id);
                }
            }

            ret.push_back(new GeometryNode(
                std::move(vertices), std::move(indices), std::move(normals),
                std::move(texcoords), std::move(material_ids)));
        }

        return ret;
    }

    static CameraNode* load_gltf_camera(const tinygltf::Camera& camera)
    {
        spdlog::info("loading camera: {}", camera.name);

        if (camera.type != "perspective")
        {
            throw std::runtime_error("only perspective camera is supported");
        }

        CameraNode* ret = new CameraNode();
        ret->set_fov(camera.perspective.yfov);
        ret->set_aspect_ratio(camera.perspective.aspectRatio);

        return ret;
    }

    static void load_gltf_node(const tinygltf::Node& node,
                               const tinygltf::Model& model, SceneNode* parent,
                               uint32_t material_id_offset)
    {
        spdlog::info("loading node: {}", node.name);

        glm::mat4 transform = load_gltf_transform(node);

        SceneNode* scene_node = nullptr;
        if (node.mesh != -1)
        {
            const std::vector<GeometryNode*> geometries = load_gltf_mesh(
                model.meshes[node.mesh], model, material_id_offset);

            for (const auto& geometry : geometries)
            {
                geometry->set_transform(transform);
                parent->add_children(geometry);
            }
        }
        else if (node.camera != -1)
        {
            CameraNode* camera = load_gltf_camera(model.cameras[node.camera]);
            camera->set_transform(transform);
            parent->add_children(camera);
        }
        else
        {
            scene_node = new SceneNode();
            scene_node->set_transform(transform);
            parent->add_children(scene_node);

            // load children
            for (const auto& child_idx : node.children)
            {
                const auto& child = model.nodes[child_idx];
                load_gltf_node(child, model, scene_node, material_id_offset);
            }
        }
    }
};

}  // namespace fredholm