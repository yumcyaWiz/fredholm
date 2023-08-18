#pragma once
#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <vector>

#include "glm/glm.hpp"
#include "glm/gtx/hash.hpp"
#include "scene.h"

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
    static void load_obj(const std::filesystem::path& filepath,
                         SceneGraph& scene_graph)
    {
        GeometryNode* geometry = new GeometryNode;
        *geometry = load_obj(filepath);

        TransformNode* transform = new TransformNode;
        transform->add_children(geometry);

        scene_graph.set_root(transform);
    }

    static void load_gltf(const std::filesystem::path& filepath,
                          SceneGraph& scene_graph)
    {
    }

   private:
    static GeometryNode load_obj(const std::filesystem::path& filepath)
    {
        std::vector<float3> m_vertices = {};
        std::vector<uint3> m_indices = {};
        std::vector<float3> m_normals = {};
        std::vector<float2> m_texcoords = {};
        std::vector<Material> m_materials = {};
        std::vector<uint> m_material_ids = {};
        std::vector<Texture> m_textures = {};

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
                unique_textures[filepath] = m_textures.size();
                // load texture
                m_textures.push_back(
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
                mat.base_color_texture_id = unique_textures[m.diffuse_texname];
            }

            // specular color
            mat.specular_color =
                make_float3(m.specular[0], m.specular[1], m.specular[2]);

            // specular color(texture)
            if (!m.specular_texname.empty())
            {
                mat.specular_color_texture_id =
                    unique_textures[m.specular_texname];
            }

            // specular roughness
            if (m.roughness > 0.0f) { mat.specular_roughness = m.roughness; }

            // specular roughness(texture)
            if (!m.roughness_texname.empty())
            {
                mat.specular_roughness_texture_id =
                    unique_textures[m.roughness_texname];
            }

            // metalness
            mat.metalness = m.metallic;

            // metalness(texture)
            if (!m.metallic_texname.empty())
            {
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
                mat.heightmap_texture_id = unique_textures[m.bump_texname];
            }

            // normal map texture
            if (!m.normal_texname.empty())
            {
                mat.normalmap_texture_id = unique_textures[m.normal_texname];
            }

            // alpha texture
            if (!m.alpha_texname.empty())
            {
                mat.alpha_texture_id = unique_textures[m.alpha_texname];
            }

            m_materials.push_back(mat);
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

                const int material_id = shapes[s].mesh.material_ids[f];
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
        spdlog::info("# of materials: {}", m_materials.size());

        return GeometryNode(std::move(m_vertices), std::move(m_indices),
                            std::move(m_normals), std::move(m_texcoords),
                            std::move(m_materials), std::move(m_material_ids));
    }
};

}  // namespace fredholm