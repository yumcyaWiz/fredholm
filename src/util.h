#pragma once
#ifndef __CUDACC__
#include "glm/glm.hpp"
#include "shared.h"

namespace fredholm
{

inline Matrix3x4 create_mat3x4_from_glm(const glm::mat4& m)
{
    return make_mat3x4(make_float4(m[0][0], m[1][0], m[2][0], m[3][0]),
                       make_float4(m[0][1], m[1][1], m[2][1], m[3][1]),
                       make_float4(m[0][2], m[1][2], m[2][2], m[3][2]));
}

}  // namespace fredholm

#endif