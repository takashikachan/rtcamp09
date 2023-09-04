#pragma once

struct LightSample
{
    float3 sample_pos;
    float3 dir;
    float3 origin;
    float3 emission;
    float distance;
    float pdf;
};

static __host__ __device__ __inline__ float3 SphericalToCartesian(float theta, float phi)
{
    float y = sin(theta);
    float x = cos(theta) * cos(phi);
    float z = cos(theta) * sin(phi);
    return make_float3(x, y, z);
}

static __host__ __device__ __inline__ void SampleLight(const AnyLight& light, float3 rand_value, LightSample& sample, float3 origin)
{
    if(light.type == SphereLight)
    {
        float3 o = origin;
        float3 c = light.position;
        float  r = light.radius;
        float3 w = c - o;
        float distance = length(w);
        w = normalize(w);
        float q = sqrt(1.0f - (r / distance) * (r / distance));
        Onb onb(w);
        float r0 = rand_value.x;
        float r1 = rand_value.y;
        float theta = acos(1 - r0 + r0 * q);
        float phi   = 2.0f * M_PIf * r1;
        float3 dir = SphericalToCartesian(theta, phi);
        onb.inverse_transform(dir);
        sample.dir = normalize(dir);
        sample.origin = origin;
        sample.emission = light.emission;
        sample.distance = distance;
        sample.pdf = 1.0f / (2.0f * M_PIf * (1.0f - q));

    }
}