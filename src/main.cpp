#include "renderer.h"

int main()
{
    fredholm::cuda_check(cuInit(0));

    fredholm::Renderer renderer;
    return 0;
}