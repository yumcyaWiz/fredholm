#pragma once
#include <optix.h>
#include <optix_stack_size.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "spdlog/spdlog.h"
//
#include "cwl/buffer.h"
#include "cwl/texture.h"
#include "cwl/util.h"
//
#include "optwl/optwl.h"
//
#include "fredholm/arhosek.h"
#include "fredholm/camera.h"
#include "fredholm/io.h"
#include "fredholm/scene.h"
#include "fredholm/shared.h"

namespace fredholm
{

class Renderer
{
 public:
  Renderer(const OptixDeviceContext& context) : m_context(context)
  {
#ifdef NDEBUG
    m_enable_validation_mode = false;
#else
    m_enable_validation_mode = true;
#endif

    CUDA_CHECK(cudaStreamCreate(&m_stream));
  }

  ~Renderer() noexcept(false)
  {
    // release framebuffer data
    if (m_d_sample_count) { m_d_sample_count.reset(); }

    // release directional light
    if (m_d_directional_light) { m_d_directional_light.reset(); }

    // release IBL
    if (m_d_ibl) { m_d_ibl.reset(); }

    // release Arhosek sky
    if (m_d_arhosek) { m_d_arhosek.reset(); }

    // release scene data
    if (m_d_vertices) { m_d_vertices.reset(); }
    if (m_d_indices) { m_d_indices.reset(); }
    if (m_d_normals) { m_d_normals.reset(); }
    if (m_d_texcoords) { m_d_texcoords.reset(); }
    if (m_d_material_ids) { m_d_material_ids.reset(); }
    if (m_d_materials) { m_d_materials.reset(); }

    if (m_d_textures.size() > 0) {
      for (auto& texture : m_d_textures) { texture.reset(); }
    }
    if (m_d_texture_headers) { m_d_texture_headers.reset(); }
    if (m_d_lights) { m_d_lights.reset(); }

    if (m_d_object_to_world) { m_d_object_to_world.reset(); }
    if (m_d_world_to_object) { m_d_world_to_object.reset(); }

    // release GAS
    for (auto& gas_output_buffer : m_gas_output_buffers) {
      if (gas_output_buffer) { gas_output_buffer.reset(); }
    }

    // release instances
    if (m_instances) { m_instances.reset(); }

    // release IAS
    if (m_ias_output_buffer) { m_ias_output_buffer.reset(); }

    // release SBT records
    if (m_d_raygen_records) { m_d_raygen_records.reset(); }
    if (m_d_miss_records) { m_d_miss_records.reset(); }
    if (m_d_hit_group_records) { m_d_hit_group_records.reset(); }

    // release pipeline
    if (m_pipeline) { OPTIX_CHECK(optixPipelineDestroy(m_pipeline)); }

    // release program groups
    if (m_raygen_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_raygen_group));
    }

    if (m_radiance_miss_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_radiance_miss_group));
    }
    if (m_shadow_miss_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_shadow_miss_group));
    }
    if (m_light_miss_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_light_miss_group));
    }

    if (m_radiance_hit_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_radiance_hit_group));
    }
    if (m_shadow_hit_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_shadow_hit_group));
    }
    if (m_light_hit_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(m_light_hit_group));
    }

    // release module
    if (m_module) { OPTIX_CHECK(optixModuleDestroy(m_module)); }

    CUDA_CHECK(cudaStreamDestroy(m_stream));
  }

  void create_module(const std::filesystem::path& ptx_filepath)
  {
    spdlog::info("[Renderer] creating OptiX module");

    OptixModuleCompileOptions options = {};
    options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    options.debugLevel = m_enable_validation_mode
                             ? OPTIX_COMPILE_DEBUG_LEVEL_FULL
                             : OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    options.optLevel = m_enable_validation_mode
                           ? OPTIX_COMPILE_OPTIMIZATION_LEVEL_0
                           : OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

    spdlog::info("[Renderer] loading {}", ptx_filepath.generic_string());
    const std::vector<char> ptx = read_file(ptx_filepath);

    // TODO: move these outside this function as much as possible
    m_pipeline_compile_options.usesMotionBlur = false;
    m_pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipeline_compile_options.numPayloadValues = 3;
    m_pipeline_compile_options.numAttributeValues = 3;
    m_pipeline_compile_options.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    if (m_enable_validation_mode) {
      m_pipeline_compile_options.exceptionFlags =
          OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
          OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    } else {
      m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        m_context, &options, &m_pipeline_compile_options, ptx.data(),
        ptx.size(), log, &sizeof_log, &m_module));
  }

  // TODO: take module and entry function name from outside?
  void create_program_group()
  {
    spdlog::info("[Renderer] creating OptiX program groups");

    OptixProgramGroupOptions options = {};

    // create raygen program group
    OptixProgramGroupDesc raygen_program_group_desc = {};
    raygen_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_program_group_desc.raygen.module = m_module;
    raygen_program_group_desc.raygen.entryFunctionName = "__raygen__rg";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(m_context, &raygen_program_group_desc, 1,
                                &options, log, &sizeof_log, &m_raygen_group));

    // create miss program group
    OptixProgramGroupDesc miss_program_group_desc = {};
    miss_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_program_group_desc.miss.module = m_module;
    miss_program_group_desc.miss.entryFunctionName = "__miss__radiance";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, &miss_program_group_desc,
                                            1, &options, log, &sizeof_log,
                                            &m_radiance_miss_group));

    miss_program_group_desc.miss.entryFunctionName = "__miss__shadow";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, &miss_program_group_desc,
                                            1, &options, log, &sizeof_log,
                                            &m_shadow_miss_group));

    miss_program_group_desc.miss.entryFunctionName = "__miss__light";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, &miss_program_group_desc,
                                            1, &options, log, &sizeof_log,
                                            &m_light_miss_group));

    // create hitgroup program group
    OptixProgramGroupDesc hitgroup_program_group_desc = {};
    hitgroup_program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_program_group_desc.hitgroup.moduleCH = m_module;
    hitgroup_program_group_desc.hitgroup.entryFunctionNameCH =
        "__closesthit__radiance";
    hitgroup_program_group_desc.hitgroup.moduleAH = m_module;
    hitgroup_program_group_desc.hitgroup.entryFunctionNameAH =
        "__anyhit__radiance";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &hitgroup_program_group_desc, 1, &options, log, &sizeof_log,
        &m_radiance_hit_group));

    hitgroup_program_group_desc.hitgroup.entryFunctionNameCH =
        "__closesthit__shadow";
    hitgroup_program_group_desc.hitgroup.entryFunctionNameAH =
        "__anyhit__shadow";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &hitgroup_program_group_desc, 1, &options, log, &sizeof_log,
        &m_shadow_hit_group));

    hitgroup_program_group_desc.hitgroup.entryFunctionNameCH =
        "__closesthit__light";
    hitgroup_program_group_desc.hitgroup.entryFunctionNameAH =
        "__anyhit__light";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &hitgroup_program_group_desc, 1, &options, log, &sizeof_log,
        &m_light_hit_group));
  }

  void create_pipeline()
  {
    spdlog::info("[Renderer] creating OptiX pipeline");

    OptixProgramGroup program_groups[] = {
        m_raygen_group,     m_radiance_miss_group, m_shadow_miss_group,
        m_light_miss_group, m_radiance_hit_group,  m_shadow_hit_group,
        m_light_hit_group};

    // create pipeline
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = m_max_trace_depth;
    pipeline_link_options.debugLevel = m_enable_validation_mode
                                           ? OPTIX_COMPILE_DEBUG_LEVEL_FULL
                                           : OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_context, &m_pipeline_compile_options, &pipeline_link_options,
        program_groups, sizeof(program_groups) / sizeof(program_groups[0]), log,
        &sizeof_log, &m_pipeline));

    // set pipeline stack size
    OptixStackSizes stack_sizes = {};
    for (auto& program_group : program_groups) {
      OPTIX_CHECK(optixUtilAccumulateStackSizes(program_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes, m_max_trace_depth, 0, 0,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));

    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        m_max_traversable_depth));
  }

  void create_sbt()
  {
    // clear previous sbt records
    m_miss_records.clear();
    m_hit_group_records.clear();

    spdlog::info("[Renderer] creating OptiX shader binding table");

    // fill raygen header
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_group, &m_raygen_record));

    // radiance miss record
    MissSbtRecord miss_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_miss_group, &miss_record));
    m_miss_records.push_back(miss_record);

    // shadow miss record
    miss_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_shadow_miss_group, &miss_record));
    m_miss_records.push_back(miss_record);

    // light miss record
    miss_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_light_miss_group, &miss_record));
    m_miss_records.push_back(miss_record);

    // fill hitgroup record
    for (size_t submesh_idx = 0; submesh_idx < m_scene.m_submesh_offsets.size();
         ++submesh_idx) {
      const uint submesh_offset = m_scene.m_submesh_offsets[submesh_idx];
      const uint n_faces = m_scene.m_submesh_n_faces[submesh_idx];

      // radiance hitgroup record
      HitGroupSbtRecord hit_record = {};
      hit_record.data.indices = m_d_indices->get_device_ptr() + submesh_offset;
      hit_record.data.material_ids =
          m_d_material_ids->get_device_ptr() + submesh_offset;
      OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_hit_group, &hit_record));
      m_hit_group_records.push_back(hit_record);

      // shadow hitgroup record
      OPTIX_CHECK(optixSbtRecordPackHeader(m_shadow_hit_group, &hit_record));
      m_hit_group_records.push_back(hit_record);

      // light hitgroup record
      OPTIX_CHECK(optixSbtRecordPackHeader(m_light_hit_group, &hit_record));
      m_hit_group_records.push_back(hit_record);
    }

    // allocate SBT records on device
    std::vector<RayGenSbtRecord> raygen_sbt_records = {m_raygen_record};
    m_d_raygen_records =
        std::make_unique<cwl::CUDABuffer<RayGenSbtRecord>>(raygen_sbt_records);
    m_d_miss_records =
        std::make_unique<cwl::CUDABuffer<MissSbtRecord>>(m_miss_records);
    m_d_hit_group_records =
        std::make_unique<cwl::CUDABuffer<HitGroupSbtRecord>>(
            m_hit_group_records);

    // fill SBT
    m_sbt.raygenRecord =
        reinterpret_cast<CUdeviceptr>(m_d_raygen_records->get_device_ptr());

    m_sbt.missRecordBase =
        reinterpret_cast<CUdeviceptr>(m_d_miss_records->get_device_ptr());
    m_sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    m_sbt.missRecordCount = m_miss_records.size();

    m_sbt.hitgroupRecordBase =
        reinterpret_cast<CUdeviceptr>(m_d_hit_group_records->get_device_ptr());
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    m_sbt.hitgroupRecordCount = m_hit_group_records.size();
  }

  void load_scene(const std::filesystem::path& filepath, bool clear = true)
  {
    spdlog::info("[Renderer] loading scene");

    m_scene.load_model(filepath, clear);
    if (!m_scene.is_valid()) { throw std::runtime_error("invalid scene"); }

    m_d_vertices =
        std::make_unique<cwl::CUDABuffer<float3>>(m_scene.m_vertices);
    m_d_indices = std::make_unique<cwl::CUDABuffer<uint3>>(m_scene.m_indices);
    m_d_normals = std::make_unique<cwl::CUDABuffer<float3>>(m_scene.m_normals);
    m_d_texcoords =
        std::make_unique<cwl::CUDABuffer<float2>>(m_scene.m_texcoords);
    m_d_material_ids =
        std::make_unique<cwl::CUDABuffer<uint>>(m_scene.m_material_ids);
    m_d_materials =
        std::make_unique<cwl::CUDABuffer<Material>>(m_scene.m_materials);

    m_d_textures.resize(m_scene.m_textures.size());
    for (int i = 0; i < m_scene.m_textures.size(); ++i) {
      const auto& tex = m_scene.m_textures[i];
      m_d_textures[i] = std::make_unique<cwl::CUDATexture<uchar4>>(
          tex.m_width, tex.m_height, tex.m_data.data(),
          tex.m_texture_type == TextureType::COLOR);
    }

    std::vector<TextureHeader> texture_headers(m_d_textures.size());
    for (int i = 0; i < m_d_textures.size(); ++i) {
      texture_headers[i].size = m_d_textures[i]->get_size();
      texture_headers[i].texture_object = m_d_textures[i]->get_texture_object();
    }
    m_d_texture_headers =
        std::make_unique<cwl::CUDABuffer<TextureHeader>>(texture_headers);

    std::vector<AreaLight> lights;
    for (int face_idx = 0; face_idx < m_scene.m_material_ids.size();
         ++face_idx) {
      const uint material_id = m_scene.m_material_ids[face_idx];
      const Material& m = m_scene.m_materials[material_id];
      if (m.emission_color.x > 0 || m.emission_color.y > 0 ||
          m.emission_color.z > 0 || m.emission_texture_id != -1) {
        AreaLight light;
        light.indices = m_scene.m_indices[face_idx];
        light.material_id = m_scene.m_material_ids[face_idx];
        light.instance_idx = m_scene.m_instance_ids[face_idx];
        lights.push_back(light);
      }
    }
    m_d_lights = std::make_unique<cwl::CUDABuffer<AreaLight>>(lights);

    std::vector<Matrix3x4> object_to_world(m_scene.m_transforms.size());
    std::vector<Matrix3x4> world_to_object(m_scene.m_transforms.size());
    for (int i = 0; i < m_scene.m_transforms.size(); ++i) {
      const auto& m = m_scene.m_transforms[i];
      const auto m_inv = glm::inverse(m);
      object_to_world[i] =
          make_mat3x4(make_float4(m[0][0], m[1][0], m[2][0], m[3][0]),
                      make_float4(m[0][1], m[1][1], m[2][1], m[3][1]),
                      make_float4(m[0][2], m[1][2], m[2][2], m[3][2]));
      world_to_object[i] = make_mat3x4(
          make_float4(m_inv[0][0], m_inv[1][0], m_inv[2][0], m_inv[3][0]),
          make_float4(m_inv[0][1], m_inv[1][1], m_inv[2][1], m_inv[3][1]),
          make_float4(m_inv[0][2], m_inv[1][2], m_inv[2][2], m_inv[3][2]));
    }
    m_d_object_to_world =
        std::make_unique<cwl::CUDABuffer<Matrix3x4>>(object_to_world);
    m_d_world_to_object =
        std::make_unique<cwl::CUDABuffer<Matrix3x4>>(world_to_object);

    spdlog::info("[Renderer] number of vertices: {}",
                 m_d_indices->get_size() * 3);
    spdlog::info("[Renderer] number of faces: {}", m_d_indices->get_size());
    spdlog::info("[Renderer] number of materials: {}",
                 m_d_materials->get_size());
    spdlog::info("[Renderer] number of textures: {}", m_d_textures.size());
    spdlog::info("[Renderer] number of lights: {}", m_d_lights->get_size());
    spdlog::info("[Renderer] number of transforms: {}",
                 m_d_object_to_world->get_size());
  }

  void build_gas()
  {
    spdlog::info("[Renderer] creating OptiX GAS, OptiX IAS");

    // GAS build option
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const uint32_t n_submeshes = m_scene.m_submesh_offsets.size();

    // NOTE: need this, since vertexBuffers take a pointer to array of device
    // pointers
    const CUdeviceptr vertex_buffer =
        reinterpret_cast<CUdeviceptr>(m_d_vertices->get_device_ptr());

    const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

    // build GAS for each submesh
    m_gas_handles.resize(n_submeshes);
    m_gas_output_buffers.resize(n_submeshes);
    for (int submesh_idx = 0; submesh_idx < n_submeshes; ++submesh_idx) {
      const uint indices_offset = m_scene.m_submesh_offsets[submesh_idx];
      const uint n_faces = m_scene.m_submesh_n_faces[submesh_idx];

      // GAS input
      OptixBuildInput input = {};
      input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      input.triangleArray.numVertices = m_d_vertices->get_size();
      input.triangleArray.vertexStrideInBytes = sizeof(float3);
      input.triangleArray.vertexBuffers = &vertex_buffer;

      input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      input.triangleArray.numIndexTriplets = n_faces;
      input.triangleArray.indexStrideInBytes = sizeof(uint3);
      input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(
          m_d_indices->get_device_ptr() + indices_offset);

      input.triangleArray.flags = flags;
      input.triangleArray.numSbtRecords = 1;

      // compute GAS buffer size
      OptixAccelBufferSizes gas_buffer_sizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &options, &input, 1,
                                               &gas_buffer_sizes));

      // build GAS
      cwl::CUDABuffer<uint8_t> gas_temp_buffer(
          gas_buffer_sizes.tempSizeInBytes);
      m_gas_output_buffers[submesh_idx] =
          std::make_unique<cwl::CUDABuffer<uint8_t>>(
              gas_buffer_sizes.outputSizeInBytes);
      OPTIX_CHECK(optixAccelBuild(
          m_context, 0, &options, &input, 1,
          reinterpret_cast<CUdeviceptr>(gas_temp_buffer.get_device_ptr()),
          gas_buffer_sizes.tempSizeInBytes,
          reinterpret_cast<CUdeviceptr>(
              m_gas_output_buffers[submesh_idx]->get_device_ptr()),
          gas_buffer_sizes.outputSizeInBytes, &m_gas_handles[submesh_idx],
          nullptr, 0));
    }
  }

  void build_ias()
  {
    const uint32_t n_submeshes = m_scene.m_submesh_offsets.size();

    // IAS build option
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // create instances
    std::vector<OptixInstance> instances(n_submeshes);
    for (int submesh_idx = 0; submesh_idx < n_submeshes; ++submesh_idx) {
      OptixInstance instance = {};

      // identify matrix
      const glm::mat4& mat = m_scene.m_transforms[submesh_idx];
      float transform[] = {mat[0][0], mat[1][0], mat[2][0], mat[3][0],
                           mat[0][1], mat[1][1], mat[2][1], mat[3][1],
                           mat[0][2], mat[1][2], mat[2][2], mat[3][2]};
      memcpy(instance.transform, transform, sizeof(float) * 12);

      instance.instanceId = submesh_idx;
      instance.sbtOffset =
          static_cast<unsigned int>(RayType::RAY_TYPE_COUNT) * submesh_idx;
      instance.visibilityMask = 1;
      instance.flags = OPTIX_INSTANCE_FLAG_NONE;
      instance.traversableHandle = m_gas_handles[submesh_idx];

      instances[submesh_idx] = instance;
    }
    m_instances = std::make_unique<cwl::CUDABuffer<OptixInstance>>(instances);

    // build single IAS
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances =
        reinterpret_cast<CUdeviceptr>(m_instances->get_device_ptr());
    input.instanceArray.numInstances = n_submeshes;

    // compute IAS buffer size
    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &options, &input, 1,
                                             &ias_buffer_sizes));

    // build IAS
    cwl::CUDABuffer<uint8_t> ias_temp_buffer(ias_buffer_sizes.tempSizeInBytes);
    m_ias_output_buffer = std::make_unique<cwl::CUDABuffer<uint8_t>>(
        ias_buffer_sizes.outputSizeInBytes);
    OPTIX_CHECK(optixAccelBuild(
        m_context, 0, &options, &input, 1,
        reinterpret_cast<CUdeviceptr>(ias_temp_buffer.get_device_ptr()),
        ias_buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(m_ias_output_buffer->get_device_ptr()),
        ias_buffer_sizes.outputSizeInBytes, &m_ias_handle, nullptr, 0));
  }

  void set_directional_light(const float3& le, const float3& dir, float angle)
  {
    spdlog::info("[Renderer] set directional light");

    DirectionalLight light;
    light.le = le;
    light.dir = normalize(dir);
    light.angle = angle;

    m_d_sun_direction = normalize(dir);

    m_d_directional_light =
        std::make_unique<cwl::DeviceObject<DirectionalLight>>(light);
  }

  void set_sky_intensity(float sky_intensity)
  {
    m_d_sky_intensity = sky_intensity;
  }

  void load_ibl(const std::filesystem::path& filepath)
  {
    spdlog::info("[Renderer] load IBL");

    const FloatTexture ibl = FloatTexture(filepath);
    m_d_ibl = std::make_unique<cwl::CUDATexture<float4>>(
        ibl.m_width, ibl.m_height, ibl.m_data.data());
  }

  void clear_ibl()
  {
    if (m_d_ibl) { m_d_ibl.reset(); }
  }

  void load_arhosek_sky(float turbidity, float albedo)
  {
    spdlog::info("[Renderer] init Arhosek sky");

    const auto cartesian_to_spherical = [](const float3& w) {
      float2 ret;
      ret.x = acosf(clamp(w.y, -1.0f, 1.0f));
      ret.y = atan2f(w.z, w.x);
      if (ret.y < 0) ret.y += 2.0f * M_PIf;
      return ret;
    };
    float elevation = cartesian_to_spherical(m_d_sun_direction).x;
    elevation = 0.5f * M_PI - elevation;

    ArHosekSkyModelState state =
        arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo, elevation);

    m_d_arhosek =
        std::make_unique<cwl::DeviceObject<ArHosekSkyModelState>>(state);
  }

  void clear_arhosek_sky()
  {
    if (m_d_arhosek) { m_d_arhosek.reset(); }
  }

  void set_time(float time)
  {
    m_scene.update_animation(time);

    // rebuild IAS
    build_ias();

    // update device transform matrix
    std::vector<Matrix3x4> object_to_world(m_scene.m_transforms.size());
    std::vector<Matrix3x4> world_to_object(m_scene.m_transforms.size());
    for (int i = 0; i < m_scene.m_transforms.size(); ++i) {
      const auto& m = m_scene.m_transforms[i];
      const auto m_inv = glm::inverse(m);
      object_to_world[i] =
          make_mat3x4(make_float4(m[0][0], m[1][0], m[2][0], m[3][0]),
                      make_float4(m[0][1], m[1][1], m[2][1], m[3][1]),
                      make_float4(m[0][2], m[1][2], m[2][2], m[3][2]));
      world_to_object[i] = make_mat3x4(
          make_float4(m_inv[0][0], m_inv[1][0], m_inv[2][0], m_inv[3][0]),
          make_float4(m_inv[0][1], m_inv[1][1], m_inv[2][1], m_inv[3][1]),
          make_float4(m_inv[0][2], m_inv[1][2], m_inv[2][2], m_inv[3][2]));
    }
    m_d_object_to_world =
        std::make_unique<cwl::CUDABuffer<Matrix3x4>>(object_to_world);
    m_d_world_to_object =
        std::make_unique<cwl::CUDABuffer<Matrix3x4>>(world_to_object);
  }

  void set_resolution(uint32_t width, uint32_t height)
  {
    m_width = width;
    m_height = height;

    init_render_states();
  }

  void init_render_states()
  {
    // init sample count buffer
    m_d_sample_count =
        std::make_unique<cwl::CUDABuffer<uint>>(m_width * m_height, 0);
  }

  void render(const Camera& camera, const float3& bg_color,
              const RenderLayer& render_layer, uint32_t n_samples,
              uint32_t max_depth)
  {
    LaunchParams params;
    params.render_layer = render_layer;
    params.sample_count = m_d_sample_count->get_device_ptr();
    params.seed = 1;

    params.width = m_width;
    params.height = m_height;
    params.n_samples = n_samples;
    params.max_depth = max_depth;

    if (m_scene.m_has_camera_transform) {
      const glm::mat4& m = m_scene.m_camera_transform;
      params.camera.transform =
          make_mat3x4(make_float4(m[0][0], m[1][0], m[2][0], m[3][0]),
                      make_float4(m[0][1], m[1][1], m[2][1], m[3][1]),
                      make_float4(m[0][2], m[1][2], m[2][2], m[3][2]));
    } else {
      params.camera.transform = make_mat3x4(
          make_float4(camera.m_transform[0][0], camera.m_transform[1][0],
                      camera.m_transform[2][0], camera.m_transform[3][0]),
          make_float4(camera.m_transform[0][1], camera.m_transform[1][1],
                      camera.m_transform[2][1], camera.m_transform[3][1]),
          make_float4(camera.m_transform[0][2], camera.m_transform[1][2],
                      camera.m_transform[2][2], camera.m_transform[3][2]));
    }
    params.camera.fov = camera.m_fov;
    params.camera.F = camera.m_F;
    params.camera.focus = camera.m_focus;

    params.object_to_world = m_d_object_to_world->get_device_ptr();
    params.world_to_object = m_d_world_to_object->get_device_ptr();

    params.vertices = m_d_vertices->get_device_ptr();
    params.normals = m_d_normals->get_device_ptr();
    params.texcoords = m_d_texcoords->get_device_ptr();

    params.materials = m_d_materials->get_device_ptr();
    params.textures = m_d_texture_headers->get_device_ptr();
    params.lights = m_d_lights->get_device_ptr();
    params.n_lights = m_d_lights->get_size();

    if (m_d_directional_light) {
      params.directional_light = m_d_directional_light->get_device_ptr();
    } else {
      params.directional_light = nullptr;
    }

    params.bg_color = bg_color;

    params.sky_intensity = m_d_sky_intensity;
    if (m_d_ibl) {
      params.ibl = m_d_ibl->get_texture_object();
    } else {
      params.ibl = 0;
    }

    params.sun_direction = m_d_sun_direction;
    if (m_d_arhosek) {
      params.arhosek = m_d_arhosek->get_device_ptr();
    } else {
      params.arhosek = nullptr;
    }

    params.ias_handle = m_ias_handle;

    // TODO: maybe this is dangerous, since optixLaunch is async?
    cwl::DeviceObject d_params(params);

    // run pipeline
    OPTIX_CHECK(
        optixLaunch(m_pipeline, m_stream,
                    reinterpret_cast<CUdeviceptr>(d_params.get_device_ptr()),
                    sizeof(LaunchParams), &m_sbt, m_width, m_height, 1));
  }

  void wait_for_completion() { CUDA_SYNC_CHECK(); }

  static void log_callback(unsigned int level, const char* tag,
                           const char* message, void* cbdata)
  {
    if (level == 4) {
      spdlog::info("[Renderer][{}] {}", tag, message);
    } else if (level == 3) {
      spdlog::warn("[Renderer][{}] {}", tag, message);
    } else if (level == 2) {
      spdlog::error("[Renderer][{}] {}", tag, message);
    } else if (level == 1) {
      spdlog::critical("[Renderer][{}] {}", tag, message);
    }
  }

 private:
  uint32_t m_width = 0;
  uint32_t m_height = 0;

  bool m_enable_validation_mode = false;

  uint32_t m_max_traversable_depth = 2;
  uint32_t m_max_trace_depth = 2;

  // scene data on host
  Scene m_scene;

  // scene data on device
  std::unique_ptr<cwl::CUDABuffer<float3>> m_d_vertices = nullptr;
  std::unique_ptr<cwl::CUDABuffer<uint3>> m_d_indices = nullptr;
  std::unique_ptr<cwl::CUDABuffer<float3>> m_d_normals = nullptr;
  std::unique_ptr<cwl::CUDABuffer<float2>> m_d_texcoords = nullptr;
  std::unique_ptr<cwl::CUDABuffer<uint>> m_d_material_ids = nullptr;

  std::unique_ptr<cwl::CUDABuffer<Material>> m_d_materials = nullptr;

  std::vector<std::unique_ptr<cwl::CUDATexture<uchar4>>> m_d_textures = {};
  std::unique_ptr<cwl::CUDABuffer<TextureHeader>> m_d_texture_headers = {};

  std::unique_ptr<cwl::CUDABuffer<AreaLight>> m_d_lights = {};

  std::unique_ptr<cwl::CUDABuffer<Matrix3x4>> m_d_object_to_world = {};
  std::unique_ptr<cwl::CUDABuffer<Matrix3x4>> m_d_world_to_object = {};

  // optix handles
  CUstream m_stream = 0;
  OptixDeviceContext m_context = 0;

  std::vector<OptixTraversableHandle> m_gas_handles = {};
  std::vector<std::unique_ptr<cwl::CUDABuffer<uint8_t>>> m_gas_output_buffers =
      {};

  std::unique_ptr<cwl::CUDABuffer<OptixInstance>> m_instances = {};
  OptixTraversableHandle m_ias_handle = {};
  std::unique_ptr<cwl::CUDABuffer<uint8_t>> m_ias_output_buffer = nullptr;

  OptixModule m_module = 0;

  OptixProgramGroup m_raygen_group = 0;

  OptixProgramGroup m_radiance_miss_group = 0;
  OptixProgramGroup m_shadow_miss_group = 0;
  OptixProgramGroup m_light_miss_group = 0;

  OptixProgramGroup m_radiance_hit_group = 0;
  OptixProgramGroup m_shadow_hit_group = 0;
  OptixProgramGroup m_light_hit_group = 0;

  OptixPipelineCompileOptions m_pipeline_compile_options = {};
  OptixPipeline m_pipeline = 0;

  OptixShaderBindingTable m_sbt = {};

  // SBT records on host
  RayGenSbtRecord m_raygen_record = {};
  std::vector<MissSbtRecord> m_miss_records = {};
  std::vector<HitGroupSbtRecord> m_hit_group_records = {};

  // SBT records on device
  std::unique_ptr<cwl::CUDABuffer<RayGenSbtRecord>> m_d_raygen_records =
      nullptr;
  std::unique_ptr<cwl::CUDABuffer<MissSbtRecord>> m_d_miss_records = nullptr;
  std::unique_ptr<cwl::CUDABuffer<HitGroupSbtRecord>> m_d_hit_group_records =
      nullptr;

  // LaunchParams data on device
  std::unique_ptr<cwl::DeviceObject<DirectionalLight>> m_d_directional_light;
  float m_d_sky_intensity = 1.0f;
  std::unique_ptr<cwl::CUDATexture<float4>> m_d_ibl;
  float3 m_d_sun_direction = make_float3(0.0f, 1.0f, 0.0f);
  std::unique_ptr<cwl::DeviceObject<ArHosekSkyModelState>> m_d_arhosek;
  std::unique_ptr<cwl::CUDABuffer<uint>> m_d_sample_count;

  static std::vector<char> read_file(const std::filesystem::path& filepath)
  {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("failed to open " + filepath.generic_string());
    }

    const size_t file_size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(file_size);

    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();

    return buffer;
  }
};

}  // namespace fredholm