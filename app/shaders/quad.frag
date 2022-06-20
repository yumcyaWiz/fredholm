#version 460 core

layout(std430, binding = 0) buffer layout_framebuffer {
  vec4 framebuffer[];
};

in vec2 texCoords;

out vec4 fragColor;

uniform vec2 resolution;
uniform int aov_type;

void main() {
  ivec2 xy = ivec2(texCoords * resolution);
  xy.y = int(resolution.y) - xy.y - 1;
  int idx = int(xy.x + resolution.x * xy.y);

  vec3 color = framebuffer[idx].xyz;
  // beauty
  if(aov_type == 0) {
    // gamma correction
    color = pow(color, vec3(1.0 / 2.2));
  }
  // position
  else if(aov_type == 1) {
  }
  // normal
  else if(aov_type == 2) {
    color = 0.5 * (color + 1.0);
  }
  // depth
  else if(aov_type == 3) {
  }
  // albedo
  else if(aov_type == 4) {
    // gamma correction
    color = pow(color, vec3(1.0 / 2.2));
  }

  fragColor = vec4(color, 1.0);
}