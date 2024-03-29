#pragma once
#include "fredholm/shared.h"
#include "sutil/vec_math.h"

#define CMJ_M 4
#define CMJ_N 4

using namespace fredholm;

// https://graphics.pixar.com/library/MultiJitteredSampling/

__forceinline__ __device__ unsigned int cmj_permute(unsigned int i,
                                                    unsigned int l,
                                                    unsigned int p)
{
  unsigned int w = l - 1;
  w |= w >> 1;
  w |= w >> 2;
  w |= w >> 4;
  w |= w >> 8;
  w |= w >> 16;
  do {
    i ^= p;
    i *= 0xe170893d;
    i ^= p >> 16;
    i ^= (i & w) >> 4;
    i ^= p >> 8;
    i *= 0x0929eb3f;
    i ^= p >> 23;
    i ^= (i & w) >> 1;
    i *= 1 | p >> 27;
    i *= 0x6935fa69;
    i ^= (i & w) >> 11;
    i *= 0x74dcb303;
    i ^= (i & w) >> 2;
    i *= 0x9e501cc3;
    i ^= (i & w) >> 2;
    i *= 0xc860a3df;
    i &= w;
    i ^= i >> 5;
  } while (i >= l);
  return (i + p) % l;
}

__forceinline__ __device__ float cmj_randfloat(unsigned int i, unsigned int p)
{
  i ^= p;
  i ^= i >> 17;
  i ^= i >> 10;
  i *= 0xb36534e5;
  i ^= i >> 12;
  i ^= i >> 21;
  i *= 0x93fc4795;
  i ^= 0xdf6e307f;
  i ^= i >> 17;
  i *= 1 | p >> 18;
  return i * (1.0f / 4294967808.0f);
}

__forceinline__ __device__ float2 cmj(unsigned int index, unsigned int scramble)
{
  index = cmj_permute(index, CMJ_M * CMJ_N, scramble * 0x51633e2d);
  unsigned int sx = cmj_permute(index % CMJ_M, CMJ_M, scramble * 0xa511e9b3);
  unsigned int sy = cmj_permute(index / CMJ_M, CMJ_N, scramble * 0x63d83595);
  float jx = cmj_randfloat(index, scramble * 0xa399d265);
  float jy = cmj_randfloat(index, scramble * 0x711ad6a5);
  return make_float2((index % CMJ_M + (sy + jx) / CMJ_N) / CMJ_M,
                     (index / CMJ_M + (sx + jy) / CMJ_M) / CMJ_N);
}

static __forceinline__ __device__ float2 cmj_2d(CMJState& state)
{
  const unsigned int index = state.n_spp % (CMJ_M * CMJ_N);
  const unsigned int scramble =
      xxhash32(make_uint4(state.n_spp / (CMJ_M * CMJ_N), state.image_idx,
                          state.depth, state.scramble));
  const float2 result = cmj(index, scramble);
  state.depth++;
  return result;
}

static __forceinline__ __device__ float cmj_1d(CMJState& state)
{
  return cmj_2d(state).x;
}

static __forceinline__ __device__ float3 cmj_3d(CMJState& state)
{
  return make_float3(cmj_2d(state), cmj_1d(state));
}

static __forceinline__ __device__ float4 cmj_4d(CMJState& state)
{
  return make_float4(cmj_2d(state), cmj_2d(state));
}