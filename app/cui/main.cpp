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
        fredholm::Renderer renderer;

        fredholm::Camera camera(glm::vec3(0, 1, 2));

        fredholm::SceneGraph scene;
        fredholm::SceneLoader::load_obj("CornellBox-Texture.obj", scene);

        fredholm::SceneDevice scene_device;
        scene_device.send(context, scene);

        fredholm::RenderOptions options;
        options.width = 512;
        options.height = 512;
        // fredholm::HelloStrategy strategy(context, debug);
        // fredholm::SimpleStrategy strategy(context, debug);
        fredholm::PtStrategy strategy(options, context, debug);
        renderer.set_render_strategy(&strategy);

        // render
        renderer.render(camera, scene_device);
        renderer.synchronize();
        renderer.save_image("output.png");
    }

    fredholm::optix_check(optixDeviceContextDestroy(context));

    return 0;
}