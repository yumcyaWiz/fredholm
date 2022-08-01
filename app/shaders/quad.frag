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
uniform float ISO;

vec3 linear_to_srgb(vec3 rgb) {
  rgb.x = rgb.x < 0.0031308 ? 12.92 * rgb.x : 1.055 * pow(rgb.x, 1.0f / 2.4f) - 0.055;
  rgb.y = rgb.y < 0.0031308 ? 12.92 * rgb.y : 1.055 * pow(rgb.y, 1.0f / 2.4f) - 0.055;
  rgb.z = rgb.z < 0.0031308 ? 12.92 * rgb.z : 1.055 * pow(rgb.z, 1.0f / 2.4f) - 0.055;
  return rgb;
}

vec3 reinhard(vec3 x) {
  return x / (1.0 + x);
}

vec3 reinhard2(vec3 x) {
  const float L_white = 4.0;
  return (x * (1.0 + x / (L_white * L_white))) / (1.0 + x);
}

// http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 filmic(vec3 x) {
  vec3 X = max(vec3(0.0), x - 0.004);
  return (X * (6.2 * X + 0.5)) / (X * (6.2 * X + 1.7) + 0.06);
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 aces_tone_mapping(vec3 color) {
  float a = 2.51f;
  float b = 0.03f;
  float c = 2.43f;
  float d = 0.59f;
  float e = 0.14f;
  return clamp((color*(a*color+b))/(color*(c*color+d)+e), vec3(0.0f), vec3(1.0f));
}

// Uchimura 2017, "HDR theory and practice"
// Math: https://www.desmos.com/calculator/gslcdxvipg
// Source: https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp
vec3 uchimura(vec3 x, float P, float a, float m, float l, float c, float b) {
  float l0 = ((P - m) * l) / a;
  float L0 = m - m / a;
  float L1 = m + (1.0 - m) / a;
  float S0 = m + l0;
  float S1 = m + a * l0;
  float C2 = (a * P) / (P - S1);
  float CP = -C2 / P;

  vec3 w0 = vec3(1.0 - smoothstep(0.0, m, x));
  vec3 w2 = vec3(step(m + l0, x));
  vec3 w1 = vec3(1.0 - w0 - w2);

  vec3 T = vec3(m * pow(x / m, vec3(c)) + b);
  vec3 S = vec3(P - (P - S1) * exp(CP * (x - S0)));
  vec3 L = vec3(m + a * (x - m));

  return T * w0 + L * w1 + S * w2;
}

vec3 uchimura(vec3 x) {
  const float P = 1.0;  // max display brightness
  const float a = 1.0;  // contrast
  const float m = 0.22; // linear section start
  const float l = 0.4;  // linear section length
  const float c = 1.33; // black
  const float b = 0.0;  // pedestal

  return uchimura(x, P, a, m, l, c, b);
}

vec3 uncharted2Tonemap(vec3 x) {
  float A = 0.15;
  float B = 0.50;
  float C = 0.10;
  float D = 0.20;
  float E = 0.02;
  float F = 0.30;
  float W = 11.2;
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 uncharted2(vec3 color) {
  const float W = 11.2;
  float exposureBias = 2.0;
  vec3 curr = uncharted2Tonemap(exposureBias * color);
  vec3 whiteScale = 1.0 / uncharted2Tonemap(vec3(W));
  return curr * whiteScale;
}

// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
float computeEV100(float aperture, float shutterTime, float ISO) {
  return log2(aperture * aperture / shutterTime * 100.0 / ISO);
}

float convertEV100ToExposure(float EV100) {
  float maxLuminance = 1.2 * pow(2.0, EV100);
  return 1.0f / maxLuminance;
}

void main() {
  ivec2 xy = ivec2(texCoords * resolution);
  xy.y = int(resolution.y) - xy.y - 1;
  int idx = int(xy.x + resolution.x * xy.y);

  vec3 color;
  // beauty
  if(aov_type == 0) {
    color = beauty[idx].xyz;

    float EV100 = computeEV100(1.0, 1.0, ISO);
    float exposure = convertEV100ToExposure(EV100);
    color = exposure * color;

    color = uchimura(color);
    color = linear_to_srgb(color);
  }
  // denoised
  if(aov_type == 1) {
    color = denoised[idx].xyz;

    float EV100 = computeEV100(1.0, 1.0, ISO);
    float exposure = convertEV100ToExposure(EV100);
    color = exposure * color;

    color = uchimura(color);
    color = linear_to_srgb(color);
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