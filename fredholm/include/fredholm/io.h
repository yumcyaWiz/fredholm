#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

// #include "device/texture.h"

namespace fredholm
{

// inline void write_ppm(const Texture2D<float4>& texture,
//                       const std::filesystem::path& filepath)
// {
//   std::ofstream file(filepath);
//   if (!file.is_open()) {
//     throw std::runtime_error("failed to open " + filepath.generic_string());
//   }

//   const int width = texture.get_width();
//   const int height = texture.get_height();

//   file << "P3" << std::endl;
//   file << width << " " << height << std::endl;
//   file << "255" << std::endl;
//   for (int j = 0; j < height; ++j) {
//     for (int i = 0; i < width; ++i) {
//       float4 v = texture.get_value(i, j);
//       const uint32_t R =
//           static_cast<uint32_t>(std::clamp(255.0f * v.x, 0.0f, 255.0f));
//       const uint32_t G =
//           static_cast<uint32_t>(std::clamp(255.0f * v.y, 0.0f, 255.0f));
//       const uint32_t B =
//           static_cast<uint32_t>(std::clamp(255.0f * v.z, 0.0f, 255.0f));
//       file << R << " " << G << " " << B << std::endl;
//     }
//   }

//   file.close();
// }

}  // namespace fredholm