#include "renderer.h"

int main()
{
    fredholm::cuda_check(cuInit(0));

    fredholm::Renderer renderer(0);
    return 0;
}