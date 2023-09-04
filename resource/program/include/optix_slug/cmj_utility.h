#pragma once

#define CMJ_N 16

struct CMJSeed
{
    unsigned int launch_index;
    unsigned int sample_index;
    unsigned int depth;
    unsigned int offset;
};

// https://graphics.pixar.com/library/MultiJitteredSampling/paper.pdf
static __host__ __device__ __inline__ unsigned int CalcCmjPermute(unsigned int i, unsigned int l, unsigned int p) 
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

static __host__ __device__ __inline__ float CalcCmjRandFloat(unsigned int i, unsigned int p)
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

static __host__ __device__ __inline__ float2 CalcCmjFloat2(int s, int m, int n, int p)
{
    int sx = CalcCmjPermute(s % m, m, p * 0xa511e9b3);
    int sy = CalcCmjPermute(s / m, n, p * 0x63d83595);
    float jx = CalcCmjRandFloat(s, p * 0xa399d265);
    float jy = CalcCmjRandFloat(s, p * 0x711ad6a5);
    float2 r = make_float2((s % m + (sy + jx) / n ) / m, (s / m + (sx + jy) / m) / n);
    return r;
}

static __host__ __device__ __inline__ float2 CalcCmjFloat2Aspect(int s, int N, int p, float a = 1.0f)
{
    int m = (int)sqrt(N * a);
    int n = (N + m - 1) / m;
    s = CalcCmjPermute(s, N, p * 0x51633e2d);
    int sx = CalcCmjPermute(s % m, m, p * 0x68bc21eb);
    int sy = CalcCmjPermute(s / m, n, p * 0x02e5be93);
    float jx = CalcCmjRandFloat(s, p * 0x967a889b);
    float jy = CalcCmjRandFloat(s, p * 0x368cc8b7);
    float2 r = make_float2((sx + (sy + jx) / n) / m, (s + jy) / N);
    return r;
}

static __host__ __device__ __inline__ float2 random_cmj2(CMJSeed& cmj_seed)
{
#if 1
    unsigned int index = (cmj_seed.sample_index) % CMJ_N;
    unsigned int x = (cmj_seed.sample_index) / CMJ_N;
    unsigned int scramble = xxhash32(make_uint4(x, cmj_seed.launch_index, cmj_seed.depth, cmj_seed.offset));
    const float2 result = CalcCmjFloat2Aspect(index, CMJ_N, scramble, 1.0f);
#else
    unsigned int seed = tea<4>( cmj_seed.launch_index, cmj_seed.offset + cmj_seed.sample_index );
    float2 result = make_float2( rnd( seed ), rnd( seed ) );
#endif
    cmj_seed.offset++;
    return result;
}

static __host__ __device__ __inline__ float random_cmj1(CMJSeed& cmj_seed)
{
    return random_cmj2(cmj_seed).x;
}

static __host__ __device__ __inline__ float3 random_cmj3(CMJSeed& cmj_seed)
{
    float2 v = random_cmj2(cmj_seed);
    return make_float3(v.x, v.y, random_cmj1(cmj_seed));
}

static __host__ __device__ __inline__ float4 random_cmj4(CMJSeed& cmj_seed)
{
    float2 v0 = random_cmj2(cmj_seed);
    float2 v1 = random_cmj2(cmj_seed);
    return make_float4(v0.x, v0.y, v1.x, v1.y);
}