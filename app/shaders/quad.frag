#version 460 core

layout(std430, binding = 0) buffer layout_framebuffer {
  vec4 framebuffer[];
};

in vec2 texCoords;

out vec4 fragColor;

uniform vec2 resolution;

void main() {
  ivec2 xy = ivec2(texCoords * resolution);
  xy.y = int(resolution.y) - xy.y - 1;
  int idx = int(xy.x + resolution.x * xy.y);
  fragColor = framebuffer[idx];
}