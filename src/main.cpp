#include "camera.h"
#include "device/texture.h"
#include "device/util.h"
#include "io.h"
#include "renderer.h"
#include "scene.h"
#include "shared.h"

int main()
{
  uint32_t width = 512;
  uint32_t height = 512;

  float3 cam_origin = make_float3(0.0f, 1.0f, 3.0f);
  float3 cam_forward = make_float3(0.0f, 0.0f, -1.0f);
  Camera camera(cam_origin, cam_forward);

#ifdef NDEBUG
  bool enable_validation_mode = false;
#else
  bool enable_validation_mode = true;
#endif

  try {
    Renderer renderer(512, 512, enable_validation_mode);

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

    renderer.render(camera);

    renderer.write_framebuffer_as_ppm("output.ppm");
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}