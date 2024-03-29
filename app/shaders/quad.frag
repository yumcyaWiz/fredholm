#version 460 core

layout(std430, binding = 0) buffer layout_beauty {
  vec4 beauty[];
};
layout(std430, binding = 1) buffer layout_denoised {
  vec4 denoised[];
};
layout(std430, binding = 2) buffer layout_position {
  vec4 position[];
};
layout(std430, binding = 3) buffer layout_normal {
  vec4 normal[];
};
layout(std430, binding = 4) buffer layout_depth {
  float depth[];
};
layout(std430, binding = 5) buffer layout_texcoord {
  vec4 texcoord[];
};
layout(std430, binding = 6) buffer layout_albedo {
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
  }
  // denoised
  if(aov_type == 1) {
    color = denoised[idx].xyz;
  }
  // position
  else if(aov_type == 2) {
    color = position[idx].xyz;
  }
  // normal
  else if(aov_type == 3) {
    color = normal[idx].xyz;
    color = 0.5 * (color + 1.0);
  }
  // depth
  else if(aov_type == 4) {
    color = vec3(depth[idx]);
  }
  // texcoord
  else if(aov_type == 5) {
    color = texcoord[idx].xyz;
  }
  // albedo
  else if(aov_type == 6) {
    color = albedo[idx].xyz;
  }

  fragColor = vec4(color, 1.0);
}