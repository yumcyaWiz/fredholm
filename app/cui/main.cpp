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

        fredholm::Camera camera(glm::vec3(0, 1, 2));

        fredholm::SceneGraph scene;
        fredholm::SceneLoader::load("CornellBox-Texture.obj", scene);

        fredholm::SceneDevice scene_device;
        scene_device.send(context, scene);

        fredholm::RenderOptions options;
        options.n_spp = 512;
        renderer.set_render_strategy(fredholm::RenderStrategyType::PT, options);

        // render
        renderer.render(camera, scene_device);
        renderer.synchronize();
        renderer.save_image("output.png");
    }

    fredholm::optix_check(optixDeviceContextDestroy(context));

    return 0;
}