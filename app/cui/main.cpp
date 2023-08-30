#include <optix.h>

#include "camera.h"
#include "cuda_util.h"
#include "loader.h"
#include "render_strategy/hello/hello.h"
#include "render_strategy/pt/pt.h"
#include "render_strategy/simple/simple.h"
#include "renderer.h"
#include "scene.h"

int main()
{
    // init CUDA
    fredholm::cuda_check(cuInit(0));
    fredholm::CUDADevice device(0);

#ifndef NDEBUG
    constexpr bool debug = true;
#else
    constexpr bool debug = false;
#endif

    // init OptiX
    optixInit();
    OptixDeviceContext context =
        fredholm::optix_create_context(device.get_context(), debug);

    {
        // init renderer
        fredholm::Renderer renderer(context, debug);

        fredholm::Camera camera(glm::vec3(0, 1, 2), glm::vec3(0, 0, -1));

        fredholm::SceneGraph scene;
        fredholm::SceneLoader::load("CornellBox-Texture.obj", scene);
        fredholm::CompiledScene compiled_scene = scene.compile();

        fredholm::SceneDevice scene_device;
        scene_device.send(context, compiled_scene);

        fredholm::RenderOptions options;
        options.n_spp = 16;
        renderer.set_render_strategy(fredholm::RenderStrategyType::PTMIS,
                                     options);

        fredholm::DirectionalLight directional_light;
        directional_light.le = make_float3(0, 0, 0);

        // render
        renderer.render(camera, directional_light, scene_device);
        renderer.save_image("output.png");
    }

    fredholm::optix_check(optixDeviceContextDestroy(context));

    return 0;
}