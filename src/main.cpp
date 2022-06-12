#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "device/texture.h"
#include "device/util.h"
#include "io.h"
#include "renderer.h"
#include "scene.h"
#include "shared.h"

template <typename T>
struct SbtRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

using RayGenSbtRecord = SbtRecord<int>;
using MissSbtRecord = SbtRecord<int>;
using HitGroupSbtRecord = SbtRecord<int>;

int main()
{
  uint32_t width = 512;
  uint32_t height = 512;
#ifdef NDEBUG
  bool enable_validation_mode = false;
#else
  bool enable_validation_mode = true;
#endif

  try {
    Renderer<RayGenSbtRecord, MissSbtRecord, HitGroupSbtRecord, Params>
        renderer(512, 512, enable_validation_mode);

    renderer.create_context();
    renderer.create_module(std::filesystem::path(MODULES_SOURCE_DIR) /
                           "triangle.ptx");
    renderer.create_program_group();
    renderer.create_pipeline(1, 1);

    RayGenSbtRecord raygen_sbt_record;
    std::vector<MissSbtRecord> miss_sbt_records = {MissSbtRecord{}};
    std::vector<HitGroupSbtRecord> hit_group_sbt_records = {
        HitGroupSbtRecord{}};
    renderer.create_sbt(raygen_sbt_record, miss_sbt_records,
                        hit_group_sbt_records);

    Scene scene;
    scene.load_obj("CornellBox-Original.obj");
    renderer.load_scene(scene);

    renderer.render();

    renderer.write_framebuffer_as_ppm("output.ppm");
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}