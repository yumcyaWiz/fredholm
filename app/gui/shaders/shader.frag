#version 460 core

layout(std430, binding = 0) buffer layout_image {
  vec4 image[];
};

in vec2 texCoords;

out vec4 fragColor;

uniform vec2 resolution;

void main() {
  ivec2 xy = ivec2(texCoords * resolution);
  xy.y = int(resolution.y) - xy.y - 1;
  int idx = int(xy.x + resolution.x * xy.y);

  vec3 color = image[idx].xyz;

  fragColor = vec4(color, 1.0);
}