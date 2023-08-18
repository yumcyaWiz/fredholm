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
#include "stb_image.h"
#include "tiny_obj_loader.h"
#include "util.h"

namespace fredholm
{

enum class ColorSpace
{
    SRGB,
    LINEAR,
};

class Texture
{
   public:
    Texture() {}
    Texture(const std::filesystem::path& filepath,
            const ColorSpace& color_space)
        : filepath(filepath), color_space(color_space)
    {
        // load(filepath);
    }

    std::filesystem::path get_filepath() const { return filepath; }

   private:
    // void load(const std::filesystem::path& filepath)
    // {
    //     int w, h, c;
    //     stbi_set_flip_vertically_on_load(true);
    //     unsigned char* img =
    //         stbi_load(filepath.c_str(), &w, &h, &c, STBI_rgb_alpha);
    //     if (!img)
    //     {
    //         throw std::runtime_error(std::format("failed to load texture
    //         {}\n",
    //                                              filepath.generic_string()));
    //     }

    //     width = w;
    //     height = h;

    //     data.resize(width * height);
    //     for (int i = 0; i < width * height; ++i)
    //     {
    //         data[i].x = img[4 * i + 0];
    //         data[i].y = img[4 * i + 1];
    //         data[i].z = img[4 * i + 2];
    //         data[i].w = img[4 * i + 3];
    //     }

    //     stbi_image_free(img);
    // }

    std::filesystem::path filepath = {};
    ColorSpace color_space = ColorSpace::SRGB;
};

enum class SceneNodeType
{
    NONE,
    TRANSFORM,
    GEOMETRY,
    INSTANCE,
};

class SceneNode
{
   public:
    std::string get_name() const { return name; }
    SceneNodeType get_type() const { return type; }
    std::vector<SceneNode*> get_children() const { return children; }

    void add_children(SceneNode* node) { children.push_back(node); }

   protected:
    std::string name = "SceneNode";
    SceneNodeType type = SceneNodeType::NONE;
    std::vector<SceneNode*> children = {};
};

// always internal node
class TransformNode : public SceneNode
{
   public:
    TransformNode()
    {
        name = "TransformNode";
        type = SceneNodeType::TRANSFORM;
    }

    glm::mat4 get_transform() const { return transform; }

    void set_transform(const glm::mat4& transform)
    {
        this->transform = transform;
    }

   private:
    glm::mat4 transform = glm::mat4(1.0f);
};

// always leaf node
class GeometryNode : public SceneNode
{
   public:
    GeometryNode()
    {
        name = "GeometryNode";
        type = SceneNodeType::GEOMETRY;
    }

    GeometryNode(const std::vector<float3>& vertices,
                 const std::vector<uint3>& indices,
                 const std::vector<float3>& normals,
                 const std::vector<float2>& texcoords,
                 const std::vector<Material>& materials,
                 const std::vector<uint>& material_ids,
                 const std::vector<Texture>& textures)
        : m_vertices(vertices),
          m_indices(indices),
          m_normals(normals),
          m_texcoords(texcoords),
          m_materials(materials),
          m_material_ids(material_ids),
          m_textures(textures)
    {
        name = "GeometryNode";
        type = SceneNodeType::GEOMETRY;
    }

    const std::vector<float3>& get_vertices() const { return m_vertices; }
    const std::vector<uint3>& get_indices() const { return m_indices; }
    const std::vector<float3>& get_normals() const { return m_normals; }
    const std::vector<float2>& get_texcoords() const { return m_texcoords; }
    const std::vector<Material>& get_materials() const { return m_materials; }
    const std::vector<uint>& get_material_ids() const { return m_material_ids; }
    const std::vector<Texture>& get_textures() const { return m_textures; }

   private:
    std::vector<float3> m_vertices = {};
    std::vector<uint3> m_indices = {};
    std::vector<float3> m_normals = {};
    std::vector<float2> m_texcoords = {};
    std::vector<Material> m_materials = {};
    std::vector<uint> m_material_ids = {};  // per face material ids
    std::vector<Texture> m_textures = {};
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
// TODO: maybe this can be placed inside SceneDevice?
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

    bool is_empty() const { return root == nullptr; }

    void clear()
    {
        destroy(root);
        root = nullptr;
    }

    void set_root(SceneNode* node)
    {
        clear();
        root = node;
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

        for (auto child : node->get_children()) { destroy(child); }
        delete node;
    }

    void compile(const SceneNode* node, const glm::mat4& transform,
                 CompiledScene& compiled_scene) const
    {
        if (node == nullptr) return;

        switch (node->get_type())
        {
            case SceneNodeType::TRANSFORM:
            {
                const TransformNode* transform_node =
                    static_cast<const TransformNode*>(node);

                const glm::mat4 transform_new =
                    transform * transform_node->get_transform();

                for (const auto& child : transform_node->get_children())
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
        std::cout << node->get_name() << std::endl;

        for (const auto& child : node->get_children())
        {
            print_tree(child, prefix + "│   ");
        }
    }

    SceneNode* root = nullptr;
};

// scene data on device
class SceneDevice
{
   public:
    SceneDevice() {}

    CUdeviceptr get_vertices() const { return vertices_buffer; }
    CUdeviceptr get_indices() const { return indices_buffer; }
    CUdeviceptr get_normals() const { return normals_buffer; }
    CUdeviceptr get_texcoords() const { return texcoords_buffer; }
    CUdeviceptr get_materials() const { return materials_buffer; }
    CUdeviceptr get_material_ids() const { return material_ids_buffer; }
    CUdeviceptr get_textures() const { return textures_buffer; }
    CUdeviceptr get_indices_offset() const { return indices_offset_buffer; }
    CUdeviceptr get_geometry_ids() const { return geometry_ids_buffer; }
    CUdeviceptr get_object_to_worlds() const { return object_to_world_buffer; }
    CUdeviceptr get_world_to_objects() const { return world_to_object_buffer; }

    void send(const OptixDeviceContext& context, const SceneGraph& scene_graph)
    {
        // compile scene graph
        const CompiledScene compiled_scene = scene_graph.compile();

        // build GAS
        destroy_gas();

        std::vector<GASBuildEntry> gas_build_entries;
        for (const auto& geometry : compiled_scene.geometry_nodes)
        {
            GASBuildEntry entry;

            cuda_check(
                cuMemAlloc(&entry.vertex_buffer,
                           geometry->get_vertices().size() * sizeof(float3)));
            cuda_check(cuMemcpyHtoD(
                entry.vertex_buffer, geometry->get_vertices().data(),
                geometry->get_vertices().size() * sizeof(float3)));
            entry.vertex_count = geometry->get_vertices().size();

            cuda_check(
                cuMemAlloc(&entry.index_buffer,
                           geometry->get_indices().size() * sizeof(uint3)));
            cuda_check(
                cuMemcpyHtoD(entry.index_buffer, geometry->get_indices().data(),
                             geometry->get_indices().size() * sizeof(uint3)));
            entry.index_count = geometry->get_indices().size();

            gas_build_entries.push_back(entry);
        }

        gas_build_outputs = optix_create_gas(context, gas_build_entries);

        // build IAS
        destroy_ias();

        std::vector<IASBuildEntry> ias_build_entries;
        for (int i = 0; i < compiled_scene.geometry_nodes.size(); ++i)
        {
            const auto& geometry = compiled_scene.geometry_nodes[i];
            const auto& transform = compiled_scene.geometry_transforms[i];

            IASBuildEntry entry;
            entry.gas_handle = gas_build_outputs[i].handle;

            entry.transform[0] = transform[0][0];
            entry.transform[1] = transform[1][0];
            entry.transform[2] = transform[2][0];
            entry.transform[3] = transform[3][0];
            entry.transform[4] = transform[0][1];
            entry.transform[5] = transform[1][1];
            entry.transform[6] = transform[2][1];
            entry.transform[7] = transform[3][1];
            entry.transform[8] = transform[0][2];
            entry.transform[9] = transform[1][2];
            entry.transform[10] = transform[2][2];
            entry.transform[11] = transform[3][2];

            // TODO: set appriopriate value
            entry.sbt_offset = 0;

            ias_build_entries.push_back(entry);
        }

        ias_build_output = optix_create_ias(context, ias_build_entries);

        // create global scene data
        std::vector<float3> vertices;
        std::vector<uint3> indices;
        std::vector<float3> normals;
        std::vector<float2> texcoords;
        std::vector<Material> materials;
        std::vector<uint> material_ids;
        std::vector<uint> indices_offset;           // key: geometry id
        std::vector<uint> geometry_ids;             // key: instance id(OptiX)
        std::vector<Matrix3x4> transforms;          // key: instance id(OptiX)
        std::vector<Matrix3x4> inverse_transforms;  // key: instance id(OptiX)

        for (int i = 0; i < compiled_scene.geometry_nodes.size(); ++i)
        {
            const auto& geometry = compiled_scene.geometry_nodes[i];

            indices_offset.push_back(indices.size());
            geometry_ids.push_back(i);

            vertices.insert(vertices.end(), geometry->get_vertices().begin(),
                            geometry->get_vertices().end());
            indices.insert(indices.end(), geometry->get_indices().begin(),
                           geometry->get_indices().end());
            normals.insert(normals.end(), geometry->get_normals().begin(),
                           geometry->get_normals().end());
            texcoords.insert(texcoords.end(), geometry->get_texcoords().begin(),
                             geometry->get_texcoords().end());
            materials.insert(materials.end(), geometry->get_materials().begin(),
                             geometry->get_materials().end());
            material_ids.insert(material_ids.end(),
                                geometry->get_material_ids().begin(),
                                geometry->get_material_ids().end());

            // load transforms
            transforms.push_back(
                create_mat3x4_from_glm(compiled_scene.geometry_transforms[i]));
            inverse_transforms.push_back(create_mat3x4_from_glm(
                glm::inverse(compiled_scene.geometry_transforms[i])));
        }

        for (int i = 0; i < compiled_scene.instance_nodes.size(); ++i)
        {
            const auto& instance = compiled_scene.instance_nodes[i];

            // find geometry id
            for (int j = 0; j < compiled_scene.geometry_nodes.size(); ++j)
            {
                if (compiled_scene.geometry_nodes[j] == instance->geometry)
                {
                    geometry_ids.push_back(j);
                    break;
                }
            }

            transforms.push_back(
                create_mat3x4_from_glm(compiled_scene.instance_transforms[i]));
            inverse_transforms.push_back(create_mat3x4_from_glm(
                glm::inverse(compiled_scene.instance_transforms[i])));
        }

        // load textures on device
        std::vector<TextureHeader> texture_headers;
        for (const auto& geometry : compiled_scene.geometry_nodes)
        {
            for (const auto& texture : geometry->get_textures())
            {
                TextureHeader header;

                // load texture on host
                int w, h, c;
                stbi_set_flip_vertically_on_load(true);
                unsigned char* img = stbi_load(texture.get_filepath().c_str(),
                                               &w, &h, &c, STBI_rgb_alpha);
                if (c != 4)
                {
                    throw std::runtime_error(
                        std::format("texture {} is not RGBA\n",
                                    texture.get_filepath().generic_string()));
                }
                header.width = w;
                header.height = h;

                // load texture on device
                cuda_check(cuMemAlloc(&header.data, w * h * sizeof(uint4)));
                cuda_check(
                    cuMemcpyHtoD(header.data, img, w * h * sizeof(uint4)));
                stbi_image_free(img);

                texture_headers.push_back(header);
            }
        }

        // allocate scene data on device
        destroy_scene_data();

        cuda_check(
            cuMemAlloc(&vertices_buffer, vertices.size() * sizeof(float3)));
        cuda_check(cuMemcpyHtoD(vertices_buffer, vertices.data(),
                                vertices.size() * sizeof(float3)));

        cuda_check(cuMemAlloc(&indices_buffer, indices.size() * sizeof(uint3)));
        cuda_check(cuMemcpyHtoD(indices_buffer, indices.data(),
                                indices.size() * sizeof(uint3)));

        cuda_check(
            cuMemAlloc(&normals_buffer, normals.size() * sizeof(float3)));
        cuda_check(cuMemcpyHtoD(normals_buffer, normals.data(),
                                normals.size() * sizeof(float3)));

        cuda_check(
            cuMemAlloc(&texcoords_buffer, texcoords.size() * sizeof(float2)));
        cuda_check(cuMemcpyHtoD(texcoords_buffer, texcoords.data(),
                                texcoords.size() * sizeof(float2)));

        cuda_check(
            cuMemAlloc(&materials_buffer, materials.size() * sizeof(Material)));
        cuda_check(cuMemcpyHtoD(materials_buffer, materials.data(),
                                materials.size() * sizeof(Material)));

        cuda_check(cuMemAlloc(&material_ids_buffer,
                              material_ids.size() * sizeof(uint)));
        cuda_check(cuMemcpyHtoD(material_ids_buffer, material_ids.data(),
                                material_ids.size() * sizeof(uint)));

        if (texture_headers.size() > 0)
        {
            cuda_check(cuMemAlloc(&textures_buffer, texture_headers.size() *
                                                        sizeof(TextureHeader)));
            cuda_check(
                cuMemcpyHtoD(textures_buffer, texture_headers.data(),
                             texture_headers.size() * sizeof(TextureHeader)));
            n_textures = texture_headers.size();
        }

        cuda_check(cuMemAlloc(&indices_offset_buffer,
                              indices_offset.size() * sizeof(uint)));
        cuda_check(cuMemcpyHtoD(indices_offset_buffer, indices_offset.data(),
                                indices_offset.size() * sizeof(uint)));

        cuda_check(cuMemAlloc(&geometry_ids_buffer,
                              geometry_ids.size() * sizeof(uint)));
        cuda_check(cuMemcpyHtoD(geometry_ids_buffer, geometry_ids.data(),
                                geometry_ids.size() * sizeof(uint)));

        cuda_check(cuMemAlloc(&object_to_world_buffer,
                              transforms.size() * sizeof(Matrix3x4)));
        cuda_check(cuMemcpyHtoD(object_to_world_buffer, transforms.data(),
                                transforms.size() * sizeof(Matrix3x4)));

        cuda_check(cuMemAlloc(&world_to_object_buffer,
                              inverse_transforms.size() * sizeof(Matrix3x4)));
        cuda_check(cuMemcpyHtoD(world_to_object_buffer,
                                inverse_transforms.data(),
                                inverse_transforms.size() * sizeof(Matrix3x4)));
    }

    OptixTraversableHandle get_ias_handle() const
    {
        return ias_build_output.handle;
    }

   private:
    void destroy_gas()
    {
        for (auto& gas_output : gas_build_outputs)
        {
            if (gas_output.output_buffer != 0)
            {
                cuda_check(cuMemFree(gas_output.output_buffer));
                gas_output.output_buffer = 0;
            }
            gas_output.handle = 0;
        }
    }

    void destroy_ias()
    {
        if (ias_build_output.instance_buffer != 0)
        {
            cuda_check(cuMemFree(ias_build_output.instance_buffer));
            ias_build_output.instance_buffer = 0;
        }
        if (ias_build_output.output_buffer != 0)
        {
            cuda_check(cuMemFree(ias_build_output.output_buffer));
            ias_build_output.output_buffer = 0;
        }
        ias_build_output.handle = 0;
    }

    void destroy_scene_data()
    {
        if (textures_buffer != 0)
        {
            for (int i = 0; i < n_textures; ++i)
            {
                TextureHeader* header =
                    reinterpret_cast<TextureHeader*>(textures_buffer) + i;
                if (header->data != 0)
                {
                    cuda_check(cuMemFree(header->data));
                    header->data = 0;
                }
            }
            n_textures = 0;

            cuda_check(cuMemFree(textures_buffer));
            textures_buffer = 0;
        }

        if (vertices_buffer != 0)
        {
            cuda_check(cuMemFree(vertices_buffer));
            vertices_buffer = 0;
        }
        if (indices_buffer != 0)
        {
            cuda_check(cuMemFree(indices_buffer));
            indices_buffer = 0;
        }
        if (normals_buffer != 0)
        {
            cuda_check(cuMemFree(normals_buffer));
            normals_buffer = 0;
        }
        if (texcoords_buffer != 0)
        {
            cuda_check(cuMemFree(texcoords_buffer));
            texcoords_buffer = 0;
        }
        if (materials_buffer != 0)
        {
            cuda_check(cuMemFree(materials_buffer));
            materials_buffer = 0;
        }
        if (material_ids_buffer != 0)
        {
            cuda_check(cuMemFree(material_ids_buffer));
            material_ids_buffer = 0;
        }
        if (indices_offset_buffer != 0)
        {
            cuda_check(cuMemFree(indices_offset_buffer));
            indices_offset_buffer = 0;
        }
        if (geometry_ids_buffer != 0)
        {
            cuda_check(cuMemFree(geometry_ids_buffer));
            geometry_ids_buffer = 0;
        }
        if (object_to_world_buffer != 0)
        {
            cuda_check(cuMemFree(object_to_world_buffer));
            object_to_world_buffer = 0;
        }
        if (world_to_object_buffer != 0)
        {
            cuda_check(cuMemFree(world_to_object_buffer));
            world_to_object_buffer = 0;
        }
    }

    std::vector<GASBuildOutput> gas_build_outputs = {};
    IASBuildOutput ias_build_output = {};

    CUdeviceptr vertices_buffer = 0;
    CUdeviceptr indices_buffer = 0;
    CUdeviceptr normals_buffer = 0;
    CUdeviceptr texcoords_buffer = 0;
    CUdeviceptr materials_buffer = 0;
    CUdeviceptr textures_buffer = 0;
    uint32_t n_textures = 0;
    CUdeviceptr material_ids_buffer = 0;
    CUdeviceptr indices_offset_buffer = 0;
    CUdeviceptr geometry_ids_buffer = 0;
    CUdeviceptr object_to_world_buffer = 0;
    CUdeviceptr world_to_object_buffer = 0;
};

}  // namespace fredholm