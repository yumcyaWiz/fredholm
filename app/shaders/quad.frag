#version 460 core

layout(std430, binding = 0) buffer layout_beauty {
  vec4 beauty[];
};

// NOTE: avoid padding problem
// https://community.khronos.org/t/size-of-elements-of-arrays-in-shader-storage-buffer-objects/69803/6
struct Vec3 {
  float x, y, z;
};
layout(std430, binding = 1) buffer layout_position {
  Vec3 position[];
};
layout(std430, binding = 2) buffer layout_normal {
  Vec3 normal[];
};

layout(std430, binding = 3) buffer layout_depth {
  float depth[];
};
layout(std430, binding = 4) buffer layout_albedo {
  vec4 albedo[];
};

in vec2 texCoords;

out vec4 fragColor;

uniform vec2 resolution;
uniform int aov_type;

void main() {
  ivec2 xy = ivec2(texCoords * resolution);
  xy.y = int(resolution.y) - xy.y - 1;
  int idx = int(xy.x + resolution.x * xy.y);

  vec3 color;
  // beauty
  if(aov_type == 0) {
    color = beauty[idx].xyz;
    color = pow(color, vec3(1.0 / 2.2));
  }
  // position
  else if(aov_type == 1) {
    Vec3 v = position[idx];
    color = vec3(v.x, v.y, v.z);
  }
  // normal
  else if(aov_type == 2) {
    Vec3 v = normal[idx];
    color = 0.5 * (vec3(v.x, v.y, v.z) + 1.0);
  }
  // depth
  else if(aov_type == 3) {
    color = vec3(depth[idx]);
  }
  // albedo
  else if(aov_type == 4) {
    color = albedo[idx].xyz;
    color = pow(color, vec3(1.0 / 2.2));
  }

  fragColor = vec4(color, 1.0);
}