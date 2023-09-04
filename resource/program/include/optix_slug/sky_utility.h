#pragma once
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