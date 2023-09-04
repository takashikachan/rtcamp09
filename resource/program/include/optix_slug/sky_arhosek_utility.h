#pragma once
#include "optix_slug/sky_arhosek_internal.h"

#define TERRESTRIAL_SOLAR_RADIUS    ( ( 0.51f * M_PIf / 180.0f ) / 2.0f )
#define ENABLE_ARHOSEK_SKY 1

static __host__ __device__ __inline__  void ArHosekSkyModelCookConfiguration(float* dataset, ArHosekSkyModelConfiguration&  config, float turbidity, float albedo, float solar_elevation)
{
    float  * elev_matrix;

    int     int_turbidity = (int)turbidity;
    float  turbidity_rem = turbidity - (float)int_turbidity;

    solar_elevation = pow(solar_elevation / (M_PIf / 2.0f), (1.0f / 3.0f));

    // alb 0 low turb

    elev_matrix = dataset + ( 9 * 6 * (int_turbidity-1) );
    
    
    for( unsigned int i = 0; i < 9; ++i )
    {
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
        config.value[i] = 
        (1.0f-albedo) * (1.0f - turbidity_rem) 
        * ( pow(1.0f-solar_elevation, 5.0f) * elev_matrix[i]  + 
           5.0f  * pow(1.0f-solar_elevation, 4.0f) * solar_elevation * elev_matrix[i+9] +
           10.0f*pow(1.0f-solar_elevation, 3.0f)*pow(solar_elevation, 2.0f) * elev_matrix[i+18] +
           10.0f*pow(1.0f-solar_elevation, 2.0f)*pow(solar_elevation, 3.0f) * elev_matrix[i+27] +
           5.0f*(1.0f-solar_elevation)*pow(solar_elevation, 4.0f) * elev_matrix[i+36] +
           pow(solar_elevation, 5.0f)  * elev_matrix[i+45]);
    }

    // alb 1 low turb
    elev_matrix = dataset + (9*6*10 + 9*6*(int_turbidity-1));
    for(unsigned int i = 0; i < 9; ++i)
    {
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
        config.value[i] += 
        (albedo) * (1.0f - turbidity_rem)
        * ( pow(1.0f-solar_elevation, 5.0f) * elev_matrix[i]  + 
           5.0f  * pow(1.0f-solar_elevation, 4.0f) * solar_elevation * elev_matrix[i+9] +
           10.0f*pow(1.0f-solar_elevation, 3.0f)*pow(solar_elevation, 2.0f) * elev_matrix[i+18] +
           10.0f*pow(1.0f-solar_elevation, 2.0f)*pow(solar_elevation, 3.0f) * elev_matrix[i+27] +
           5.0f*(1.0f-solar_elevation)*pow(solar_elevation, 4.0f) * elev_matrix[i+36] +
           pow(solar_elevation, 5.0f)  * elev_matrix[i+45]);
    }

    if(int_turbidity == 10)
        return;

    // alb 0 high turb
    elev_matrix = dataset + (9*6*(int_turbidity));
    for(unsigned int i = 0; i < 9; ++i)
    {
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
        config.value[i] += 
        (1.0f-albedo) * (turbidity_rem)
        * ( pow(1.0f-solar_elevation, 5.0f) * elev_matrix[i]  + 
           5.0f  * pow(1.0f-solar_elevation, 4.0f) * solar_elevation * elev_matrix[i+9] +
           10.0f*pow(1.0f-solar_elevation, 3.0f)*pow(solar_elevation, 2.0f) * elev_matrix[i+18] +
           10.0f*pow(1.0f-solar_elevation, 2.0f)*pow(solar_elevation, 3.0f) * elev_matrix[i+27] +
           5.0f*(1.0f-solar_elevation)*pow(solar_elevation, 4.0f) * elev_matrix[i+36] +
           pow(solar_elevation, 5.0f)  * elev_matrix[i+45]);
    }

    // alb 1 high turb
    elev_matrix = dataset + (9*6*10 + 9*6*(int_turbidity));
    for(unsigned int i = 0; i < 9; ++i)
    {
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
        config.value[i] += 
        (albedo) * (turbidity_rem)
        * ( pow(1.0f-solar_elevation, 5.0f) * elev_matrix[i]  + 
           5.0f  * pow(1.0f-solar_elevation, 4.0f) * solar_elevation * elev_matrix[i+9] +
           10.0f*pow(1.0f-solar_elevation, 3.0f)*pow(solar_elevation, 2.0f) * elev_matrix[i+18] +
           10.0f*pow(1.0f-solar_elevation, 2.0f)*pow(solar_elevation, 3.0f) * elev_matrix[i+27] +
           5.0f*(1.0f-solar_elevation)*pow(solar_elevation, 4.0f) * elev_matrix[i+36] +
           pow(solar_elevation, 5.0f)  * elev_matrix[i+45]);
    }
}

static __host__ __device__ __inline__  float ArHosekSkyModelCookRadianceConfiguration(float* dataset, float turbidity, float albedo, float solar_elevation)
{
    float* elev_matrix;

    int int_turbidity = (int)turbidity;
    float turbidity_rem = turbidity - (float)int_turbidity;
    float res;
    solar_elevation = pow(solar_elevation / (M_PIf / 2.0f), (1.0f / 3.0f));

    // alb 0 low turb
    elev_matrix = dataset + (6*(int_turbidity-1));
    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    res = (1.0f-albedo) * (1.0f - turbidity_rem) *
        ( pow(1.0f-solar_elevation, 5.0f) * elev_matrix[0] +
         5.0f*pow(1.0f-solar_elevation, 4.0f)*solar_elevation * elev_matrix[1] +
         10.0f*pow(1.0f-solar_elevation, 3.0f)*pow(solar_elevation, 2.0f) * elev_matrix[2] +
         10.0f*pow(1.0f-solar_elevation, 2.0f)*pow(solar_elevation, 3.0f) * elev_matrix[3] +
         5.0f*(1.0f-solar_elevation)*pow(solar_elevation, 4.0f) * elev_matrix[4] +
         pow(solar_elevation, 5.0f) * elev_matrix[5]);

    // alb 1 low turb
    elev_matrix = dataset + (6*10 + 6*(int_turbidity-1));
    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    res += (albedo) * (1.0f - turbidity_rem) *
        ( pow(1.0f-solar_elevation, 5.0f) * elev_matrix[0] +
         5.0f*pow(1.0f-solar_elevation, 4.0f)*solar_elevation * elev_matrix[1] +
         10.0f*pow(1.0f-solar_elevation, 3.0f)*pow(solar_elevation, 2.0f) * elev_matrix[2] +
         10.0f*pow(1.0f-solar_elevation, 2.0f)*pow(solar_elevation, 3.0f) * elev_matrix[3] +
         5.0f*(1.0f-solar_elevation)*pow(solar_elevation, 4.0f) * elev_matrix[4] +
         pow(solar_elevation, 5.0f) * elev_matrix[5]);
    if(int_turbidity == 10)
        return res;

    // alb 0 high turb
    elev_matrix = dataset + (6*(int_turbidity));
    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    res += (1.0f-albedo) * (turbidity_rem) *
        ( pow(1.0f-solar_elevation, 5.0f) * elev_matrix[0] +
         5.0f*pow(1.0f-solar_elevation, 4.0f)*solar_elevation * elev_matrix[1] +
         10.0f*pow(1.0f-solar_elevation, 3.0f)*pow(solar_elevation, 2.0f) * elev_matrix[2] +
         10.0f*pow(1.0f-solar_elevation, 2.0f)*pow(solar_elevation, 3.0f) * elev_matrix[3] +
         5.0f*(1.0f-solar_elevation)*pow(solar_elevation, 4.0f) * elev_matrix[4] +
         pow(solar_elevation, 5.0f) * elev_matrix[5]);

    // alb 1 high turb
    elev_matrix = dataset + (6*10 + 6*(int_turbidity));
    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    res += (albedo) * (turbidity_rem) *
        ( pow(1.0f-solar_elevation, 5.0f) * elev_matrix[0] +
         5.0f*pow(1.0f-solar_elevation, 4.0f)*solar_elevation * elev_matrix[1] +
         10.0f*pow(1.0f-solar_elevation, 3.0f)*pow(solar_elevation, 2.0f) * elev_matrix[2] +
         10.0f*pow(1.0f-solar_elevation, 2.0f)*pow(solar_elevation, 3.0f) * elev_matrix[3] +
         5.0f*(1.0f-solar_elevation)*pow(solar_elevation, 4.0f) * elev_matrix[4] +
         pow(solar_elevation, 5.0f) * elev_matrix[5]);
    return res;
}

static __host__ __device__ __inline__ void arhosek_rgb_skymodelstate_alloc_init(ArHosekSkyModelState& state, const float turbidity, const float albedo, const float elevation)
{    
    state.solar_radius = TERRESTRIAL_SOLAR_RADIUS;
    state.turbidity    = turbidity;
    state.albedo       = albedo;
    state.elevation    = elevation;

    for( unsigned int channel = 0; channel < 3; ++channel )
    {
        ArHosekSkyModelCookConfiguration(
            datasetsRGB[channel], 
            state.configs[channel], 
            turbidity, 
            albedo, 
            elevation
            );
        
        state.radiances[channel] = 
        ArHosekSkyModelCookRadianceConfiguration(
            datasetsRGBRad[channel],
            turbidity, 
            albedo,
            elevation
            );
    }
}


static __host__ __device__ __inline__ float3 ComputeCoefficientMie(float3 lambda, float3 K, float turbidity, float jungeexp = 4.0)
{
    const float c = max(0.f, 0.6544f * turbidity - 0.6510f) * 1e-16f;
    const float mie =  0.434f * c * M_PIf * pow(2.0f*M_PIf, jungeexp - 2.0f);
    float3 tmp;
    tmp.x = pow(lambda.x, jungeexp - 2);
    tmp.y = pow(lambda.x, jungeexp - 2);
    tmp.z = pow(lambda.x, jungeexp - 2);

    float3 ret;
    ret.x = mie * K.x / tmp.x;
    ret.y = mie * K.y / tmp.y;
    ret.z = mie * K.z / tmp.z;

    return ret;
}

static __host__ __device__ __inline__ float3 ComputeCoefficientRayleigh(float3 lambda)
{
    const float n = 1.0003f;
    const float N = 2.545e25f;
    const float p = 0.035f;

    float3 l4;
    l4.x = lambda.x * lambda.x * lambda.x * lambda.x * 3.0f * N;
    l4.y = lambda.y * lambda.y * lambda.y * lambda.y * 3.0f * N;
    l4.z = lambda.z * lambda.z * lambda.z * lambda.z * 3.0f * N;

    float a = 8.0f * M_PIf * M_PIf * M_PIf * pow(n * n - 1.0f, 2.0f);
    float c = (6.0f + 3.0f * p) / (6.0f - 7.0f * p);
    
    float3 ret;
    ret.x = a / l4.x * c;
    ret.y = a / l4.y * c;
    ret.z = a / l4.z * c;
    return ret;
}