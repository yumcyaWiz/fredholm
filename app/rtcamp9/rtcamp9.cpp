#include <optix.h>

#include <chrono>
#include <mutex>
#include <queue>

#include "camera.h"
#include "cuda_util.h"
#include "loader.h"
#include "render_strategy/hello/hello.h"
#include "render_strategy/pt/pt.h"
#include "render_strategy/simple/simple.h"
#include "renderer.h"
#include "scene.h"
#include "stb_image_write.h"

class Timer
{
   public:
    Timer() {}

    void start() { m_start = std::chrono::steady_clock::now(); }

    void end() { m_end = std::chrono::steady_clock::now(); }

    template <typename T>
    int elapsed() const
    {
        return std::chrono::duration_cast<T>(std::chrono::steady_clock::now() -
                                             m_start)
            .count();
    }

    template <typename T>
    int duration() const
    {
        return std::chrono::duration_cast<T>(m_end - m_start).count();
    }

   private:
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_end;
};

int main()
{
    constexpr uint32_t width = 1920;
    constexpr uint32_t height = 1080;
    constexpr uint32_t n_spp = 16;
    constexpr float max_time = 10.0f;
    constexpr float fps = 24.0f;
    constexpr float time_step = 1.0f / fps;
    constexpr float kill_time = 295.0f;
    const std::filesystem::path filepath =
        "../resources/scenes/ai58/"
        "AI58_009.gltf";

    Timer global_timer;
    global_timer.start();

    constexpr bool debug = false;

    {
        std::queue<std::pair<int, const float4*>> queue;
        std::mutex queue_mutex;
        bool render_finished = false;

        // rendering loop
        std::thread render_thread(
            [&]
            {
                // init CUDA
                fredholm::cuda_check(cuInit(0));
                fredholm::CUDADevice device(0);

                // init OptiX
                optixInit();
                OptixDeviceContext context =
                    fredholm::optix_create_context(device.get_context(), debug);

                {
                    // init renderer
                    fredholm::Renderer renderer(context, debug);

                    fredholm::Camera camera(glm::vec3(0, 1, 2),
                                            glm::vec3(0, 0, -1));
                    camera.set_fov(0.25f * M_PI);

                    Timer scene_load_timer;
                    scene_load_timer.start();

                    fredholm::SceneGraph scene;
                    fredholm::SceneLoader::load(filepath, scene);
                    fredholm::CompiledScene compiled_scene = scene.compile();

                    fredholm::SceneDevice scene_device;
                    scene_device.send(context, compiled_scene);

                    scene_load_timer.end();
                    spdlog::info(
                        "scene_load_time: {} ms",
                        scene_load_timer.duration<std::chrono::milliseconds>());

                    renderer.set_render_strategy(
                        fredholm::RenderStrategyType::PTMIS);

                    float3 sun_direction = normalize(make_float3(1, 1, 1));
                    fredholm::DirectionalLight directional_light;
                    directional_light.le = make_float3(30, 20, 10);
                    directional_light.dir = sun_direction;
                    directional_light.angle = 30.0f;

                    int frame_idx = 0;
                    float time = 0.0f;

                    bool camera_jump1 = true;
                    bool camera_jump2 = true;
                    bool camera_jump3 = true;

                    while (true)
                    {
                        // spdlog::info("rendering frame: {}", frame_idx);
                        // spdlog::info("current frame time: {}", time);

                        Timer frame_timer;
                        frame_timer.start();

                        if (time > max_time ||
                            global_timer.elapsed<std::chrono::seconds>() >
                                kill_time)
                        {
                            render_finished = true;
                            break;
                        }

                        renderer.clear_render();

                        // update camera
                        if (time < 3.0f)
                        {
                            if (camera_jump1)
                            {
                                camera_jump1 = false;
                                camera.set_origin(glm::vec3(-478, 151, 769));
                                camera.set_forward(
                                    glm::vec3(0.83, -0.008, -0.54));
                            }
                            camera.move(fredholm::CameraMovement::FORWARD,
                                        1.0f);
                        }
                        else if (time < 5.0f)
                        {
                            if (camera_jump2)
                            {
                                camera_jump2 = false;
                                camera.set_origin(glm::vec3(1122, 168, 658));
                                camera.set_forward(
                                    glm::vec3(-0.95, -0.10, -0.27));
                            }
                            camera.move(fredholm::CameraMovement::RIGHT, 1.0f);
                        }
                        else if (time < max_time)
                        {
                            if (camera_jump3)
                            {
                                camera_jump3 = false;
                                camera.set_origin(glm::vec3(-434, 148, 200));
                                camera.set_forward(
                                    glm::vec3(0.83, -0.008, -0.54));
                            }
                            camera.move(fredholm::CameraMovement::RIGHT, 1.0f);
                        }

                        // update sun direction
                        sun_direction = normalize(make_float3(
                            1.0f, std::cos(0.5f * time + 2.0f) + 1.1f,
                            std::sin(0.5f * time + 2.0f)));
                        directional_light.dir = sun_direction;

                        // update sky
                        scene_device.update_arhosek(1.0f, sun_direction, 3.0f,
                                                    0.3f);

                        // render
                        Timer render_timer;
                        render_timer.start();
                        {
                            renderer.render(camera, directional_light,
                                            scene_device);
                            renderer.synchronize();
                        }
                        render_timer.end();
                        // spdlog::info(
                        //     "render_time: {} ms",
                        //     render_timer.duration<std::chrono::milliseconds>());

                        // copy image from device to host
                        Timer image_transfer_timer;
                        image_transfer_timer.start();
                        float4* image_f4 = new float4[width * height];
                        renderer.get_aov(fredholm::AOVType::FINAL)
                            .copy_d_to_h(image_f4);
                        image_transfer_timer.end();

                        // push image to queue
                        {
                            std::lock_guard<std::mutex> lock(queue_mutex);
                            queue.push({frame_idx, image_f4});
                        }

                        // go to next frame
                        frame_idx++;
                        time += time_step;

                        frame_timer.end();
                        spdlog::info(
                            "frame {}: {} ms", frame_idx,
                            frame_timer.duration<std::chrono::milliseconds>());
                    }
                }
            });

        std::vector<uchar4> image_c4(width * height);

        std::thread save_thread(
            [&]
            {
                while (true)
                {
                    if (global_timer.elapsed<std::chrono::seconds>() >
                        kill_time)
                    {
                        break;
                    }
                    if (render_finished && queue.empty()) { break; }

                    if (queue.empty()) continue;

                    // pop image from queue
                    int frame_idx;
                    const float4* image_f4 = nullptr;
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        frame_idx = queue.front().first;
                        image_f4 = queue.front().second;
                        queue.pop();
                    }

                    Timer image_convert_time;
                    image_convert_time.start();
                    {
                        for (int j = 0; j < height; ++j)
                        {
                            for (int i = 0; i < width; ++i)
                            {
                                const int idx = i + width * j;
                                const float4& v = image_f4[idx];
                                image_c4[idx].x = static_cast<unsigned char>(
                                    std::clamp(255.0f * v.x, 0.0f, 255.0f));
                                image_c4[idx].y = static_cast<unsigned char>(
                                    std::clamp(255.0f * v.y, 0.0f, 255.0f));
                                image_c4[idx].z = static_cast<unsigned char>(
                                    std::clamp(255.0f * v.z, 0.0f, 255.0f));
                                image_c4[idx].w = 255;
                            }
                        }
                    }
                    delete[] image_f4;
                    image_convert_time.end();
                    // spdlog::info("image_convert_time: {} ms",
                    //              image_convert_time
                    //                  .duration<std::chrono::milliseconds>());

                    Timer image_write_timer;
                    image_write_timer.start();
                    {
                        const std::string filename =
                            std::format("output/{:03d}.png", frame_idx);
                        stbi_write_png(filename.c_str(), width, height, 4,
                                       image_c4.data(), sizeof(uchar4) * width);
                        spdlog::info("image {} is written", filename);
                    }
                    image_write_timer.end();
                    // spdlog::info("image_write_time: {} ms",
                    //              image_write_timer
                    //                  .duration<std::chrono::milliseconds>());
                }
            });

        render_thread.join();
        save_thread.join();
    }

    return 0;
}