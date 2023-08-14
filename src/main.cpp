#include "renderer.h"

int main()
{
    fredholm::cuda_check_error(cuInit(0));

    fredholm::Renderer renderer(0);
    return 0;
}