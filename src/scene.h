#pragma once
#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <vector>

#include "camera.h"
#include "glm/glm.hpp"
#include "helper_math.h"
#include "image_io.h"
#include "shared.h"
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
    }

    std::filesystem::path get_filepath() const { return filepath; }

   private:
    std::filesystem::path filepath = {};
    ColorSpace color_space = ColorSpace::SRGB;
};

enum class SceneNodeType
{
    DEFAULT,
    GEOMETRY,
    INSTANCE,
    CAMERA,
};

class SceneNode
{
   public:
    std::string get_name() const { return name; }

    SceneNodeType get_type() const { return type; }

    glm::mat4 get_transform() const { return transform; }
    void set_transform(const glm::mat4& transform)
    {
        this->transform = transform;
    }

    std::vector<SceneNode*> get_children() const { return children; }

    void add_children(SceneNode* node) { children.push_back(node); }

   protected:
    std::string name = "SceneNode";
    SceneNodeType type = SceneNodeType::DEFAULT;
    glm::mat4 transform = glm::identity<glm::mat4>();
    std::vector<SceneNode*> children = {};
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

    GeometryNode(const std::vector<float3>&& vertices,
                 const std::vector<uint3>&& indices,
                 const std::vector<float3>&& normals,
                 const std::vector<float2>&& texcoords,
                 const std::vector<uint>&& material_ids)
        : m_vertices(vertices),
          m_indices(indices),
          m_normals(normals),
          m_texcoords(texcoords),
          m_material_ids(material_ids)
    {
        name = "GeometryNode";
        type = SceneNodeType::GEOMETRY;
    }

    const std::vector<float3>& get_vertices() const { return m_vertices; }
    const std::vector<uint3>& get_indices() const { return m_indices; }
    const std::vector<float3>& get_normals() const { return m_normals; }
    const std::vector<float2>& get_texcoords() const { return m_texcoords; }
    const std::vector<uint>& get_material_ids() const { return m_material_ids; }

   private:
    std::vector<float3> m_vertices = {};
    std::vector<uint3> m_indices = {};
    std::vector<float3> m_normals = {};
    std::vector<float2> m_texcoords = {};
    std::vector<uint> m_material_ids = {};  // per face material ids
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

// always leaf node
struct CameraNode : public SceneNode
{
    CameraNode()
    {
        name = "CameraNode";
        type = SceneNodeType::CAMERA;
    }

    void set_fov(const float fov) { this->fov = fov; }
    void set_aspect_ratio(const float aspect_ratio)
    {
        this->aspect_ratio = aspect_ratio;
    }

    float fov = 0.0f;
    float aspect_ratio = 1.0f;
};

struct Animation
{
    SceneNode* target_node = nullptr;

    std::vector<float> translation_times = {};
    std::vector<glm::vec3> translation_values = {};

    std::vector<float> rotation_times = {};
    std::vector<glm::quat> rotation_values = {};

    std::vector<float> scale_times = {};
    std::vector<glm::vec3> scale_values = {};

    bool is_valid() const { return target_node != nullptr; }

    bool has_translation() const { return translation_times.size() > 0; }
    bool has_rotation() const { return rotation_times.size() > 0; }
    bool has_scale() const { return scale_times.size() > 0; }
};

// used for creating GAS and IAS
// TODO: maybe this can be placed inside SceneDevice?
struct CompiledScene
{
    std::vector<const GeometryNode*> geometry_nodes = {};
    std::vector<glm::mat4> geometry_transforms = {};
    std::vector<const InstanceNode*> instance_nodes = {};
    std::vector<glm::mat4> instance_transforms = {};

    std::vector<Material> m_materials = {};
    std::vector<Texture> m_textures = {};

    Camera camera = {};

    Texture envmap = {};
};

// TODO: add lights
class SceneGraph
{
   public:
    SceneGraph() {}

    ~SceneGraph() { clear(); }

    bool is_empty() const { return root_nodes.size() == 0; }

    uint32_t n_materials() const { return m_materials.size(); }
    uint32_t n_textures() const { return m_textures.size(); }

    void clear()
    {
        for (const auto& root : root_nodes) { destroy(root); }
        root_nodes.clear();

        m_materials.clear();
        m_textures.clear();

        m_animations.clear();

        envmap = {};
    }

    void add_root(SceneNode* node) { root_nodes.push_back(node); }

    void add_material(const Material& material)
    {
        m_materials.push_back(material);
    }

    void add_texture(const Texture& texture) { m_textures.push_back(texture); }

    const Texture& get_envmap() const { return envmap; }
    void set_envmap(const Texture& texture) { envmap = texture; }
    bool has_envmap() const
    {
        return envmap.get_filepath().generic_string().size() > 0;
    }

    CompiledScene compile() const
    {
        CompiledScene ret;
        for (const auto& root : root_nodes)
        {
            compile_nodes(root, glm::identity<glm::mat4>(), ret);
        }
        ret.m_materials = m_materials;
        ret.m_textures = m_textures;
        ret.envmap = envmap;
        return ret;
    }

    void print_tree() const
    {
        for (const auto& root : root_nodes) { print_tree(root, ""); }
    }

    void update_animation(const float time)
    {
        for (const auto& animation : m_animations)
        {
            if (!animation.is_valid()) continue;

            glm::vec3 transform = {0.0f, 0.0f, 0.0f};
            if (animation.has_translation())
            {
                transform = animation_lerp(animation.translation_times,
                                           animation.translation_values, time);
            }

            glm::quat rotation = {1.0f, 0.0f, 0.0f, 0.0f};
            if (animation.has_rotation())
            {
                rotation = animation_lerp(animation.rotation_times,
                                          animation.rotation_values, time);
            }

            glm::vec3 scale = {1.0f, 1.0f, 1.0f};
            if (animation.has_scale())
            {
                scale = animation_lerp(animation.scale_times,
                                       animation.scale_values, time);
            }

            const glm::mat4 transform_new =
                glm::translate(glm::identity<glm::mat4>(), transform) *
                glm::mat4_cast(rotation) *
                glm::scale(glm::identity<glm::mat4>(), scale);

            animation.target_node->set_transform(transform_new);
        }
    }

   private:
    template <typename T>
    static T animation_lerp(const std::vector<float>& times,
                            const std::vector<T>& values, const float time)
    {
        const float t = std::fmod(time, times.back());
        const int idx1 =
            std::lower_bound(times.begin(), times.end(), t) - times.begin();
        const int idx0 = std::max(idx1 - 1, 0);

        const float h = (t - times[idx0]) / (times[idx1] - times[idx0]);
        const T value0 = values[idx0];
        const T value1 = values[idx1];
        return glm::mix(value0, value1, h);
    }

    void destroy(SceneNode* node)
    {
        if (node == nullptr) return;

        for (auto child : node->get_children()) { destroy(child); }
        delete node;
    }

    void compile_nodes(const SceneNode* node, const glm::mat4& transform,
                       CompiledScene& compiled_scene) const
    {
        if (node == nullptr) return;

        const glm::mat4 transform_new = transform * node->get_transform();

        switch (node->get_type())
        {
            case SceneNodeType::DEFAULT:
            {
                for (const auto& child : node->get_children())
                {
                    compile_nodes(child, transform_new, compiled_scene);
                }
                break;
            }
            case SceneNodeType::GEOMETRY:
            {
                const GeometryNode* geometry_node =
                    static_cast<const GeometryNode*>(node);
                compiled_scene.geometry_nodes.push_back(geometry_node);
                compiled_scene.geometry_transforms.push_back(transform_new);
                break;
            }
            case SceneNodeType::INSTANCE:
            {
                const InstanceNode* instance_node =
                    static_cast<const InstanceNode*>(node);

                compiled_scene.instance_nodes.push_back(instance_node);
                compiled_scene.instance_transforms.push_back(transform_new);
                break;
            }
            case SceneNodeType::CAMERA:
            {
                const CameraNode* camera_node =
                    static_cast<const CameraNode*>(node);

                // TODO: handle multiple camera
                // TODO: set aspect ratio
                compiled_scene.camera.set_transform(transform_new);
                compiled_scene.camera.set_fov(camera_node->fov);

                return;
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

    std::vector<SceneNode*> root_nodes = {};

    std::vector<Material> m_materials = {};
    std::vector<Texture> m_textures = {};

    Texture envmap = {};

    std::vector<Animation> m_animations = {};
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
    CUdeviceptr get_n_vertices_buffer() const { return n_vertices_buffer; }
    CUdeviceptr get_n_faces_buffer() const { return n_faces_buffer; }
    CUdeviceptr get_geometry_ids() const { return geometry_ids_buffer; }
    CUdeviceptr get_object_to_worlds() const { return object_to_world_buffer; }
    CUdeviceptr get_world_to_objects() const { return world_to_object_buffer; }

    CUdeviceptr get_area_lights() const { return area_lights_buffer; }

    uint2 get_envmap_resolution() const { return envmap_resolution; }
    CUdeviceptr get_envmap() const { return envmap_buffer; }

    uint32_t get_n_vertices() const { return n_vertices; }
    uint32_t get_n_faces() const { return n_faces; }
    uint32_t get_n_materials() const { return n_materials; }
    uint32_t get_n_geometries() const { return n_geometries; }
    uint32_t get_n_instances() const { return n_instances; }
    uint32_t get_n_textures() const { return n_textures; }
    uint32_t get_n_area_lights() const { return n_area_lights; }

    void update_transforms(const OptixDeviceContext& context,
                           const CompiledScene& compiled_scene)
    {
        // TODO: do not recreate IAS
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

        // TODO: update only affected transforms
        std::vector<Matrix3x4> transforms;
        std::vector<Matrix3x4> inverse_transforms;
        for (int i = 0; i < compiled_scene.geometry_nodes.size(); ++i)
        {
            transforms.push_back(
                create_mat3x4_from_glm(compiled_scene.geometry_transforms[i]));
            inverse_transforms.push_back(create_mat3x4_from_glm(
                glm::inverse(compiled_scene.geometry_transforms[i])));
        }
        for (int i = 0; i < compiled_scene.instance_nodes.size(); ++i)
        {
            transforms.push_back(
                create_mat3x4_from_glm(compiled_scene.instance_transforms[i]));
            inverse_transforms.push_back(create_mat3x4_from_glm(
                glm::inverse(compiled_scene.instance_transforms[i])));
        }

        cuda_check(cuMemcpyHtoD(object_to_world_buffer, transforms.data(),
                                transforms.size() * sizeof(Matrix3x4)));
        cuda_check(cuMemcpyHtoD(world_to_object_buffer,
                                inverse_transforms.data(),
                                inverse_transforms.size() * sizeof(Matrix3x4)));
    }

    void send(const OptixDeviceContext& context,
              const CompiledScene& compiled_scene)
    {
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

        // TODO: add instanced geometry after this loop(use instance_nodes)
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
        std::vector<uint> n_vertices_buffer_h;
        std::vector<uint> n_faces_buffer_h;
        std::vector<uint> geometry_ids;
        std::vector<Matrix3x4> transforms;
        std::vector<Matrix3x4> inverse_transforms;

        for (int i = 0; i < compiled_scene.geometry_nodes.size(); ++i)
        {
            const auto& geometry = compiled_scene.geometry_nodes[i];

            n_vertices_buffer_h.push_back(vertices.size());
            n_faces_buffer_h.push_back(indices.size());
            geometry_ids.push_back(i);

            vertices.insert(vertices.end(), geometry->get_vertices().begin(),
                            geometry->get_vertices().end());
            indices.insert(indices.end(), geometry->get_indices().begin(),
                           geometry->get_indices().end());
            normals.insert(normals.end(), geometry->get_normals().begin(),
                           geometry->get_normals().end());
            texcoords.insert(texcoords.end(), geometry->get_texcoords().begin(),
                             geometry->get_texcoords().end());
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

        materials = compiled_scene.m_materials;

        // load textures on device
        std::vector<TextureHeader> texture_headers;
        for (const auto& texture : compiled_scene.m_textures)
        {
            spdlog::info("loading texture: {}",
                         texture.get_filepath().generic_string());

            TextureHeader header;

            // TODO: change loading based on texture colorspace
            uint32_t width, height;
            const std::vector<uchar4> image = ImageLoader::load_ldr_image(
                texture.get_filepath(), width, height);
            header.width = width;
            header.height = height;

            // load texture on device
            cuda_check(
                cuMemAlloc(&header.data, width * height * sizeof(uchar4)));
            cuda_check(cuMemcpyHtoD(header.data, image.data(),
                                    width * height * sizeof(uchar4)));

            texture_headers.push_back(header);
        }

        spdlog::info("loading area lights");

        // load area lights
        std::vector<AreaLight> area_lights;
        for (int geom_id = 0; geom_id < compiled_scene.geometry_nodes.size();
             ++geom_id)
        {
            const uint vertices_offset = n_vertices_buffer_h[geom_id];
            const uint indices_offset = n_faces_buffer_h[geom_id];
            for (int prim_id = 0;
                 prim_id <
                 compiled_scene.geometry_nodes[geom_id]->get_indices().size();
                 ++prim_id)
            {
                const uint material_id = material_ids[indices_offset + prim_id];
                const auto& material = materials[material_id];
                if (material.has_emission())
                {
                    AreaLight light;
                    light.indices =
                        indices[indices_offset + prim_id] + vertices_offset;
                    light.material_id = material_id;
                    light.instance_idx = geom_id;
                    area_lights.push_back(light);
                }
            }
        }

        spdlog::info("===scene device info===");
        spdlog::info("# of vertices: {}", vertices.size());
        spdlog::info("# of faces: {}", indices.size());
        spdlog::info("# of materials: {}", materials.size());
        spdlog::info("# of textures: {}", texture_headers.size());
        spdlog::info("# of geometries: {}",
                     compiled_scene.geometry_nodes.size());
        spdlog::info("# of instances: {}",
                     compiled_scene.instance_nodes.size());
        spdlog::info("# of area lights: {}", area_lights.size());
        spdlog::info("======");

        // allocate scene data on device
        destroy_scene_data();

        spdlog::info("allocating scene data on device");

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

        // TODO: use uint64 for big scenes
        cuda_check(cuMemAlloc(&n_vertices_buffer,
                              n_vertices_buffer_h.size() * sizeof(uint)));
        cuda_check(cuMemcpyHtoD(n_vertices_buffer, n_vertices_buffer_h.data(),
                                n_vertices_buffer_h.size() * sizeof(uint)));

        // TODO: use uint64 for big scenes
        cuda_check(cuMemAlloc(&n_faces_buffer,
                              n_faces_buffer_h.size() * sizeof(uint)));
        cuda_check(cuMemcpyHtoD(n_faces_buffer, n_faces_buffer_h.data(),
                                n_faces_buffer_h.size() * sizeof(uint)));

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

        if (area_lights.size() > 0)
        {
            cuda_check(cuMemAlloc(&area_lights_buffer,
                                  area_lights.size() * sizeof(AreaLight)));
            cuda_check(cuMemcpyHtoD(area_lights_buffer, area_lights.data(),
                                    area_lights.size() * sizeof(AreaLight)));
            n_area_lights = area_lights.size();
        }

        spdlog::info("loading envmap");

        // load envmap
        if (compiled_scene.envmap.get_filepath().generic_string().size() > 0)
        {
            const Texture& envmap = compiled_scene.envmap;

            uint32_t width, height;
            const std::vector<float3> image = ImageLoader::load_hdr_image(
                envmap.get_filepath(), width, height);
            envmap_resolution.x = width;
            envmap_resolution.y = height;

            cuda_check(
                cuMemAlloc(&envmap_buffer, width * height * sizeof(float3)));
            cuda_check(cuMemcpyHtoD(envmap_buffer, image.data(),
                                    width * height * sizeof(float3)));
        }

        n_vertices = vertices.size();
        n_faces = indices.size();
        n_materials = materials.size();
        n_geometries = compiled_scene.geometry_nodes.size();
        n_instances = compiled_scene.instance_nodes.size();
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
                // TODO: fix segfault
                // TextureHeader* header =
                //     reinterpret_cast<TextureHeader*>(textures_buffer)
                //     + i;
                // if (header != nullptr && header->data
                // != 0)
                // {
                //     cuda_check(cuMemFree(header->data));
                //     header->data = 0;
                // }
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
        if (n_vertices_buffer != 0)
        {
            cuda_check(cuMemFree(n_vertices_buffer));
            n_vertices_buffer = 0;
        }
        if (n_faces_buffer != 0)
        {
            cuda_check(cuMemFree(n_faces_buffer));
            n_faces_buffer = 0;
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

        if (area_lights_buffer != 0)
        {
            cuda_check(cuMemFree(area_lights_buffer));
            area_lights_buffer = 0;
            n_area_lights = 0;
        }

        if (envmap_buffer != 0)
        {
            cuda_check(cuMemFree(envmap_buffer));
            envmap_buffer = 0;
            envmap_resolution = make_uint2(0, 0);
        }

        // reset statistics
        n_vertices = 0;
        n_faces = 0;
        n_materials = 0;
        n_textures = 0;
        n_geometries = 0;
        n_instances = 0;
    }

    std::vector<GASBuildOutput> gas_build_outputs = {};
    IASBuildOutput ias_build_output = {};

    CUdeviceptr vertices_buffer = 0;         // key: vertex id
    CUdeviceptr indices_buffer = 0;          // key: face id
    CUdeviceptr normals_buffer = 0;          // key: vertex id
    CUdeviceptr texcoords_buffer = 0;        // key: vertex id
    CUdeviceptr materials_buffer = 0;        // key: material id
    CUdeviceptr textures_buffer = 0;         // key: vertex id
    CUdeviceptr material_ids_buffer = 0;     // key: face id
    CUdeviceptr n_vertices_buffer = 0;       // key: geometry id
    CUdeviceptr n_faces_buffer = 0;          // key: geometry id
    CUdeviceptr geometry_ids_buffer = 0;     // key: instance id
    CUdeviceptr object_to_world_buffer = 0;  // key: instance id
    CUdeviceptr world_to_object_buffer = 0;  // key: instance id

    CUdeviceptr area_lights_buffer = 0;  // key: light id
    uint n_area_lights = 0;

    uint2 envmap_resolution = make_uint2(0, 0);
    CUdeviceptr envmap_buffer = 0;

    // statistics
    uint32_t n_vertices = 0;
    uint32_t n_faces = 0;
    uint32_t n_materials = 0;
    uint32_t n_textures = 0;
    uint32_t n_geometries = 0;
    uint32_t n_instances = 0;
};

}  // namespace fredholm