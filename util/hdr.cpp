#include <vector>

#include "helper_math.h"
#include "stb_image_write.h"

int main()
{
    std::vector<float3> img(1);
    img[0] = make_float3(1.0f, 1.0f, 1.0f);
    stbi_write_hdr("white.hdr", 1, 1, 3, reinterpret_cast<float*>(img.data()));
    return 0;
}