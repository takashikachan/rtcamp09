#pragma once
#include "optix_slug/sky_arhosek_utility.h"
static __host__ __device__ __inline__ float ArHosekSkyModel_GetRadianceInternal(ArHosekSkyModelConfiguration configuration, float theta, float gamma)
{
    float rate = cos(theta);
    rate = rate < 0 ? 0 : rate;
    const float expM = exp(configuration.value[4] * gamma);
    const float rayM = cos(gamma)*cos(gamma);
    const float mieM = (1.0f + cos(gamma)*cos(gamma)) / pow((1.0f + configuration.value[8]*configuration.value[8] - 2.0f * configuration.value[8]*cos(gamma)), 1.5f);
    const float zenith = sqrt(rate);

    return (1.0f + configuration.value[0] * exp(configuration.value[1] / (cos(theta) + 0.01f))) *
            (configuration.value[2] + configuration.value[3] * expM + configuration.value[5] * rayM + configuration.value[6] * mieM + configuration.value[7] * zenith);
}

static __host__ __device__ __inline__ float arhosek_tristim_skymodel_radiance(ArHosekSkyModelState& state, float theta, float gamma, int channel)
{
    return ArHosekSkyModel_GetRadianceInternal(state.configs[channel],  theta, gamma) * state.radiances[channel];
}

static __host__ __device__ __inline__ float3 EvaluateSky(ArHosekSkyModelState& state, float3 ray_dir, float3 ray_pos, float3 sun_dir, float sky_intensity)
{
    
    float cosTheta0 = ray_dir.x * sun_dir.x + ray_dir.y * sun_dir.y + ray_dir.z * sun_dir.z;
    float cosTheta1 = ray_dir.x * 0.0f + ray_dir.y * 1.0f + ray_dir.z * 0.0f;

    float gamma = acos(cosTheta0);
    float theta = acos(cosTheta1);
    bool lower_hemisphere = theta >= 0.5f * M_PIf;
    if(lower_hemisphere)
    {
        theta = M_PIf - theta;
    }

    float3 emitter;
    emitter.x = arhosek_tristim_skymodel_radiance(state, theta, gamma, 0) * sky_intensity;
    emitter.y = arhosek_tristim_skymodel_radiance(state, theta, gamma, 1) * sky_intensity;
    emitter.z = arhosek_tristim_skymodel_radiance(state, theta, gamma, 2) * sky_intensity;
    return emitter;
}