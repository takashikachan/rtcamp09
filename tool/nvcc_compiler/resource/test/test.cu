#include <optix/optix.h>
#include <cuda/helpers.h>

#include "slug/test0.h"
#include "slug/test1.h"

#include <sutil/vec_math.h>
extern "C" {
    __constant__ Params params;
}


static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}


static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 d = 2.0f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(dim.x),
        static_cast<float>(idx.y) / static_cast<float>(dim.y)
    ) - 1.0f;

    origin = params.cam_eye;
    direction = normalize(d.x * U + d.y * V + W);
}


extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    computeRay(idx, dim, ray_origin, ray_direction);

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.0f,
        1e16f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,
        1,
        0,
        p0, p1, p2);
    float3 result;
    result.x = __uint_as_float(p0);
    result.y = __uint_as_float(p1);
    result.z = __uint_as_float(p2);

    // Record results in our output raster
    params.image[idx.y * params.image_width + idx.x] = make_color(result);
}


extern "C" __global__ void __miss__ms()
{
    MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    setPayload(miss_data->bg_color);
}


extern "C" __global__ void __closesthit__ch()
{
    const float2 barycentrics = optixGetTriangleBarycentrics();

    setPayload(make_float3(barycentrics, 1.0f));
}
