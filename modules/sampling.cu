struct RNGState {
  ulong state = 0;
  ulong inc = 0xdeadbeef;
};

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
static __forceinline__ __device__ uint pcg32_random_r(RNGState* rng)
{
  ulong oldstate = rng->state;
  // Advance internal state
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  uint xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static __forceinline__ __device__ float frandom(RNGState* rng)
{
  return pcg32_random_r(rng) / static_cast<float>(0xffffffff);
}

static __forceinline__ __device__ float3
sample_cosine_weighted_hemisphere(const float u1, const float u2)
{
  // Uniformly sample disk.
  const float r = sqrtf(u1);
  const float phi = 2.0f * M_PIf * u2;

  float3 p;
  p.x = r * cosf(phi);
  p.y = r * sinf(phi);
  // Project up to hemisphere.
  p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));

  return p;
}