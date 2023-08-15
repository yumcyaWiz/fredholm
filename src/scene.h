#pragma once
#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <vector>

#include "glm/glm.hpp"
#include "glm/gtx/hash.hpp"
#include "helper_math.h"
#include "shared.h"
#include "spdlog/spdlog.h"
#include "tiny_obj_loader.h"

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

enum class SceneNodeType
{
    NONE,
    TRANSFORM,
    GEOMETRY,
    INSTANCE,
};

struct SceneNode
{
    void add_children(SceneNode* node) { children.push_back(node); }

    std::string name = "SceneNode";
    SceneNodeType type = SceneNodeType::NONE;
    std::vector<SceneNode*> children = {};
};

// always internal node
struct TransformNode : public SceneNode
{
    TransformNode()
    {
        name = "TransformNode";
        type = SceneNodeType::TRANSFORM;
    }

    void set_transform(const glm::mat4& transform)
    {
        this->transform = transform;
    }

    glm::mat4 transform = glm::mat4(1.0f);
};

// always leaf node
struct GeometryNode : public SceneNode
{
    GeometryNode()
    {
        name = "GeometryNode";
        type = SceneNodeType::GEOMETRY;
    }

    void load_obj(const std::filesystem::path& filepath)
    {
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
    }

    std::vector<float3> m_vertices = {};
    std::vector<uint3> m_indices = {};
    std::vector<float3> m_normals = {};
    std::vector<float3> m_tangents = {};
    std::vector<float2> m_texcoords = {};
};

// always leaf node
struct InstanceNode : public SceneNode
{
    InstanceNode()
    {
        name = "InstanceNode";
        type = SceneNodeType::INSTANCE;
    }

    const GeometryNode* geometry = nullptr;
};

// used for creating GAS and IAS
struct CompiledScene
{
    std::vector<const GeometryNode*> geometry_nodes = {};
    std::vector<glm::mat4> geometry_transforms = {};
    std::vector<const InstanceNode*> instance_nodes = {};
    std::vector<glm::mat4> instance_transforms = {};
};

class SceneGraph
{
   public:
    SceneGraph() {}

    ~SceneGraph() { destroy(root); }

    void load_obj(const std::filesystem::path& filepath)
    {
        GeometryNode* geometry = new GeometryNode;
        geometry->load_obj(filepath);

        TransformNode* transform = new TransformNode;
        transform->children.push_back(geometry);

        root = transform;
    }

    CompiledScene compile() const
    {
        CompiledScene ret;
        compile(root, glm::mat4(1.0f), ret);
        return ret;
    }

    void print_tree() const { print_tree(root, ""); }

   private:
    void destroy(SceneNode* node)
    {
        if (node == nullptr) return;

        for (auto child : node->children) { destroy(child); }
        delete node;
    }

    void compile(const SceneNode* node, const glm::mat4& transform,
                 CompiledScene& compiled_scene) const
    {
        if (node == nullptr) return;

        switch (node->type)
        {
            case SceneNodeType::TRANSFORM:
            {
                const TransformNode* transform_node =
                    static_cast<const TransformNode*>(node);

                const glm::mat4 transform_new =
                    transform * transform_node->transform;

                for (const auto& child : transform_node->children)
                {
                    compile(child, transform_new, compiled_scene);
                }
                break;
            }
            case SceneNodeType::GEOMETRY:
            {
                const GeometryNode* geometry_node =
                    static_cast<const GeometryNode*>(node);

                compiled_scene.geometry_nodes.push_back(geometry_node);
                compiled_scene.geometry_transforms.push_back(transform);
                break;
            }
            case SceneNodeType::INSTANCE:
            {
                const InstanceNode* instance_node =
                    static_cast<const InstanceNode*>(node);

                compiled_scene.instance_nodes.push_back(instance_node);
                compiled_scene.instance_transforms.push_back(transform);
                break;
            }
            default:
            {
                throw std::runtime_error("unknown scene node type");
            }
        }
    }

    void print_tree(const SceneNode* node, const std::string& prefix) const
    {
        if (node == nullptr) return;

        std::cout << prefix;
        std::cout << "├── ";
        std::cout << node->name << std::endl;

        for (const auto& child : node->children)
        {
            print_tree(child, prefix + "│   ");
        }
    }

    SceneNode* root = nullptr;
};

}  // namespace fredholm