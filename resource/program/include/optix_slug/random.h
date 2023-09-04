#pragma once

template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
{
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for( unsigned int n = 0; n < N; n++ )
  {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }

  return v0;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

static __host__ __device__ __inline__ unsigned int lcg2(unsigned int &prev)
{
  prev = (prev*8121 + 28411)  % 134456;
  return prev;
}

static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}

static __host__ __device__ __inline__ unsigned int rot_seed( unsigned int seed, unsigned int frame )
{
    return seed ^ frame;
}

static __host__ __device__ __inline__ float2 rand_normal2(unsigned int seed)
{
    make_float2(rnd(seed), rnd(seed));
}

//https://www.shadertoy.com/view/fdGyzD
static __host__ __device__ __inline__ unsigned int xorshift(unsigned int x) 
{
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

static __host__ __device__ __inline__ float rand_xorshift(unsigned int seed) 
{
    unsigned int x = xorshift(seed);
    return float(x) / float(0x0fffffff);
}

// https://www.shadertoy.com/view/Xt3cDn
static __host__ __device__ __inline__ unsigned int xxhash(uint2 p)
{
    const unsigned int PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const unsigned int PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    unsigned int h32 = p.y + PRIME32_5 + p.x*PRIME32_3;
    h32 = PRIME32_4*((h32 << 17) | (h32 >> (32 - 17))); //Initial testing suggests this line could be omitted for extra perf
    h32 = PRIME32_2*(h32^(h32 >> 15));
    h32 = PRIME32_3*(h32^(h32 >> 13));
    return h32^(h32 >> 16);
}

static __host__ __device__ __inline__ unsigned int xxhash32(const uint4& p)
{
  const unsigned int PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
  const unsigned int PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
  unsigned int h32 = p.w + PRIME32_5 + p.x * PRIME32_3;
  h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
  h32 += p.y * PRIME32_3;
  h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
  h32 += p.z * PRIME32_3;
  h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
  h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
  h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
  return h32 ^ (h32 >> 16);
}

static __host__ __device__ __inline__ float rand_xxhash(uint2 v)
{
    unsigned int n = xxhash(v);
    return float(n) / float(0xffffffff);
}

static __host__ __device__ __inline__ float2 rand_xxhash2(uint2 v)
{
    unsigned int n = xxhash(v);
    uint2 rz = make_uint2(n, n * 48271);
    float x = float((rz.x >> 1) & unsigned int(0x7fffffff)) / float(0x7fffffff);
    float y = float((rz.y >> 1) & unsigned int(0x7fffffff)) / float(0x7fffffff);
    return make_float2(x, y);
}

static __host__ __device__ __inline__ float3 rand_xxhash3(uint2 v)
{
    unsigned int n = xxhash(v);
    uint3 rz = make_uint3(n, n * 16807, n * 48271);
    float x = float((rz.x >> 1) & unsigned int(0x7fffffff)) / float(0x7fffffff);
    float y = float((rz.y >> 1) & unsigned int(0x7fffffff)) / float(0x7fffffff);
    float z = float((rz.z >> 1) & unsigned int(0x7fffffff)) / float(0x7fffffff);
    return make_float3(x, y, z);
}

static __host__ __device__ __inline__ float4 rand_xxhash4(uint2 v)
{
    unsigned int n = xxhash(v);
    uint4 rz = make_uint4(n, n * 16807, n * 48271, n * 69621);
    float x = float((rz.x >> 1) & unsigned int(0x7fffffff)) / float(0x7fffffff);
    float y = float((rz.y >> 1) & unsigned int(0x7fffffff)) / float(0x7fffffff);
    float z = float((rz.z >> 1) & unsigned int(0x7fffffff)) / float(0x7fffffff);
    float w = float((rz.w >> 1) & unsigned int(0x7fffffff)) / float(0x7fffffff);
    return make_float4(x, y, z, w);
}