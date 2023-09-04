#include <optix/optix.h>
#include <optix/internal/optix_device_impl_transformations.h>
#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include "optix_slug/default_shader_data.h"
#include "optix_slug/random.h"
#include "optix_slug/utility.h"
#include "optix_slug/cmj_utility.h"
#include "optix_slug/sky_utility.h"
#include "optix_slug/brdf_utility.h"
#include "optix_slug/light_utility.h"

extern "C" {
__constant__ LaunchParam params;
}

struct RandomSeed
{
    CMJSeed cmj_seed;
    unsigned int normal_seed;
};

struct PayloadData
{
    RandomSeed seed;            //!< 乱数seed
    int depth;                  //!< バウンス回数
    bool occluded;              //!< シャドウレイ用の遮蔽フラグ
    int done;                   //!< レイトレース終了か

    float3 albedo;              //!< アルベド値
    float3 normal;              //!< 法線
    float3 position;            //!< ワールド座標
    float3 debug;               //!< デバッグ用
    
    float3 origin;              //!< レイの始点
    float3 direction;           //!< レイの方向
    
    float3 radiance;           //!< 放射輝度
    float3 throughput;         //!< 減衰値
};

typedef union
{
    PayloadData* ptr;
    uint2 dat;
}Payload;

__forceinline__ __device__ uint2 SplitPointer(PayloadData* ptr)
{
    Payload payload;
    payload.ptr = ptr;
    return payload.dat;
}

__forceinline__ __device__ PayloadData* MergePointer(unsigned int p0, unsigned int p1)
{
  Payload payload;
  payload.dat.x = p0;
  payload.dat.y = p1;
  return payload.ptr;
}

static __forceinline__ __device__ void traceRay(OptixTraversableHandle handle, float3 ray_origin, 
                                                float3 ray_direction, float tmin, float tmax, PayloadData& prd)
{
    uint2 payload = SplitPointer(&prd);
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f, OptixVisibilityMask( 1 ), OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, payload.x,  payload.y );
}

static __forceinline__ __device__ void traceShadowRay(OptixTraversableHandle handle, float3 ray_origin, 
                                                      float3 ray_direction, float tmin, float tmax, PayloadData& prd)
{
    uint2 payload = SplitPointer(&prd);
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f, OptixVisibilityMask( 1 ), 
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, RAY_TYPE_OCCLUSION, RAY_TYPE_COUNT, RAY_TYPE_OCCLUSION, payload.x, payload.y);
}

static __forceinline__ __device__ void InitSeed(RandomSeed& seed, unsigned int launch_index, unsigned int depth, unsigned int sample_index)
{
    seed.cmj_seed.launch_index = launch_index;
    seed.cmj_seed.depth = depth;
    seed.cmj_seed.sample_index = sample_index;
    seed.cmj_seed.offset = 0;
}

static __forceinline__ __device__ BRDFMaterial InitBRDFMaterial(Material& material, float4 base_color)
{
    BRDFMaterial bsdf_material;
    float3 color  =make_float3(base_color.x, base_color.y, base_color.z);
    bsdf_material.base_color = color;
    bsdf_material.ior = material.ior;
    bsdf_material.specular_tint = material.specular_tint;
    bsdf_material.specular_trans = material.specular_trans;
    bsdf_material.sheen = material.sheen;
    bsdf_material.sheen_tint = material.sheen_tint;
    bsdf_material.roughness = material.roughness;
    bsdf_material.metallic = material.metallic;
    bsdf_material.clearcoat = material.clearcoat;
    bsdf_material.clearcoat_gloss = material.clearcoat_gloss;
    bsdf_material.subsurface = material.subsurface;
    bsdf_material.anisotropic = material.anisotropic;
    bsdf_material.debug_diffuse = material.debug_diffuse;
    bsdf_material.debug_specular = material.debug_specular;
    return bsdf_material;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 launch_index = optixGetLaunchIndex();
    const unsigned int width = params.width;
    const unsigned int height = params.height;
    const unsigned int subframe_index = params.subframe_index;
    const unsigned int launch_samples = params.samples_per_launch;
    const unsigned int max_depth = params.max_depth;
    float max_dist = params.max_dist;
    float min_dist = params.min_dist;

    const float3 eye = params.camera.eye;
    const float3 U = params.camera.U;
    const float3 V = params.camera.V;
    const float3 W = params.camera.W;

    const bool enable_russian_roulette = params.debug.russian_roulette;

    unsigned int image_index = CalcBufferIndex(width, height, launch_index, true);

    float3 albedo = make_float3(0.0f);
    float3 normal = make_float3(0.0f);
    float3 position = make_float3(0.0f);
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    
    for(unsigned int sample_id = 0; sample_id < launch_samples; sample_id ++)
    {
        PayloadData prd;
        unsigned int total_sample_index = sample_id;
        InitSeed(prd.seed, image_index, 0, total_sample_index + subframe_index);

        const float2 subpixel_jitter = random_cmj2(prd.seed.cmj_seed);
        float3 ray_dir = CalcDirecton(launch_index, subpixel_jitter, width, height, U, V, W);
        float3 ray_origin = eye;

        prd.radiance = make_float3(0.0f, 0.0f, 0.0f);
        prd.throughput = make_float3(1.0f, 1.0f, 1.0f);
        prd.depth = 0;

        while(true)
        {
            prd.seed.cmj_seed.depth = prd.depth;
            traceRay(params.handle, ray_origin, ray_dir, min_dist, max_dist, prd);            
            albedo = prd.albedo;
            normal = prd.normal;
            position = prd.position;
            
            if(prd.done || prd.depth >= max_depth)
            {
                break;
            }
            if(enable_russian_roulette)
            {
                float russian_roulette = clamp(CalculateLuminance(prd.throughput), 0.0f, 1.0f);
                float rand_value = random_cmj1(prd.seed.cmj_seed);
                if(rand_value >= russian_roulette)
                {
                    break;
                }
                prd.throughput /= russian_roulette;
            }
            ray_origin = prd.origin;
            ray_dir = prd.direction;
            ++prd.depth;
        }
        radiance += prd.radiance / launch_samples;
    }
    
    float3 result = radiance;
    /*
    if(subframe_index > 0)
    {
        const float a = 1.0f / static_cast<float>( subframe_index + 1 );
        float3 accume = make_float3(params.accum_buffer[ image_index ]);
        result = lerp(accume, result, a);
    }
    */
    params.albedo_buffer[image_index] = make_float4(albedo, 1.0f);
    params.normal_buffer[image_index] = make_float4(normal, 1.0f);
    params.position_buffer[image_index] = make_float4(position, 1.0f);
    params.frame_buffer[ image_index ] = make_float4( result, 1.0f);
    params.accum_buffer[ image_index ] = make_float4( result, 1.0f);
}

extern "C" __global__ void __miss__shadow()
{
    PayloadData* prd = MergePointer(optixGetPayload_0(), optixGetPayload_1());
    prd->occluded = false;
}

extern "C" __global__ void __miss__radiance()
{    
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    PayloadData* prd = MergePointer(optixGetPayload_0(), optixGetPayload_1());
    const float3 ray_dir = normalize(optixGetWorldRayDirection());
    const float3 ray_pos = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    if( prd->depth == 0 )
    {
        prd->albedo = make_float3(0.0f, 0.0f, 0.0f);
        prd->normal = make_float3(0.0f, 0.0f, 0.0f);
        prd->position = make_float3(0.0f, 0.0f, 0.0f);
    }
    float gamma = acos(dot(ray_dir, params.direct_light.dir));
    float theta = acos(dot(ray_dir, make_float3(0.0f, 1.0f, 0.0f)));
    bool lower_hemisphere = theta >= 0.5f * M_PIf;
    if(lower_hemisphere)
    {
        theta = M_PIf - theta;
    }
    float3 emitter = EvaluateSky(params.direct_light.sky_state, ray_dir, ray_pos, params.direct_light.dir, params.direct_light.sky_intensity);
    prd->radiance += emitter * prd->throughput;
    prd->done = true;
}

extern "C" __global__ void __closesthit__shadow()
{
    // HitGoupデータを取得
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    // 三角形IDを取得し、index値を参集
    const int prim_idx = optixGetPrimitiveIndex();
    
    // materialを取得し、specu_transの値をみて影にするか判断
    unsigned int material_id = rt_data->material_ids[prim_idx];
    Material material = params.materials[material_id];
    BRDFMaterial bsdf_material = InitBRDFMaterial(material, make_float4(0.0f, 0.0f, 0.0f, 0.0f));

    PayloadData* prd = MergePointer(optixGetPayload_0(), optixGetPayload_1());
    if(bsdf_material.specular_trans > 0.0f)
    {
        prd->occluded = false;
    }else
    {
        prd->occluded = true;
    }
}

extern "C" __global__ void __closesthit__radiance()
{
    // HitGoupデータを取得
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    // 三角形IDを取得し、index値を参集
    const int prim_idx = optixGetPrimitiveIndex();
    int index_idx_offset = prim_idx * 3;

    // 三角形上のUV値を取得
    const float2 trim_uv = make_float2(optixGetTriangleBarycentrics().x, optixGetTriangleBarycentrics().y);

    // SBTから座標、法線、面法線、接線、テクスチャのUV座標を取得
    float3 position = make_float3(0.0f, 0.0f, 0.0f);
    float3 normal = make_float3(0.0f, 0.0f, 0.0f);
    float3 tanget = make_float3(0.0f, 0.0f, 0.0f);
    float3 binormal = make_float3(0.0f, 0.0f, 0.0f);
    float3 face_normal = make_float3(0.0f, 0.0f, 0.0f);
    unsigned int material_id = 0;
    float2 uv = make_float2(0.0f, 0.0f);
    {
        // index値を算出
        const unsigned int index0 = rt_data->indices[ index_idx_offset + 0 ];
        const unsigned int index1 = rt_data->indices[ index_idx_offset + 1 ];
        const unsigned int index2 = rt_data->indices[ index_idx_offset + 2 ];

        // 座標を算出
        const float3 p0 = rt_data->vertices[index0];
        const float3 p1 = rt_data->vertices[index1];
        const float3 p2 = rt_data->vertices[index2];
        position = CalcTraiangleSample(p0, p1, p2, trim_uv);

        // 法線を算出
        const float3 n0 = rt_data->normals[index0];
        const float3 n1 = rt_data->normals[index1];
        const float3 n2 = rt_data->normals[index2];
        normal = CalcTraiangleSample(n0, n1, n2, trim_uv);

        tanget = cross(normal, make_float3(0.0f, 1.0f, 0.0));
        binormal = cross(tanget, normal);

        // 面法線を算出
        face_normal = normalize( cross( p1 - p0, p2 - p0 ) );

        // テクスチャのUV座標を算出
        const float2 t0 = rt_data->texcoords[index0];
        const float2 t1 = rt_data->texcoords[index1];
        const float2 t2 = rt_data->texcoords[index2];
        uv = CalcTraiangleSample(t0, t1, t2, trim_uv);

        //uv.x = 1.0f - uv.x;
        uv.y = 1.0f - uv.y;
        material_id = rt_data->material_ids[prim_idx];
    }

    Material material = params.materials[material_id];

    // テクスチャフェッチ(アルベド)
    float4 base_color = make_float4(material.base_color[0], material.base_color[1], material.base_color[2], 1.0f);
    if(material.albedo > 0)
    {
        base_color = TexSample(material.albedo, uv, false);
    }
    
    // テクスチャフェッチ(BumpMap)
    float3 tex_normal = make_float3(0.0f, 1.0f, 0.0f);
    if(material.bump > 0)
    {
        float2 offset[4];
        offset[0] = make_float2(0.0f / 512.0f, -1.0f / 512.0f);
        offset[1] = make_float2(0.0f / 512.0f, 1.0f / 512.0f);
        offset[2] = make_float2(-1.0f / 512.0f, 0.0f / 512.0f);
        offset[3] = make_float2(1.0f / 512.0f, 0.0f / 512.0f);

        float delta = 2.0f;
        float bump0 = TexSample(material.bump, uv + offset[0], false).x;
        float bump1 = TexSample(material.bump, uv + offset[1], false).x;
        float bump2 = TexSample(material.bump, uv + offset[2], false).x;
        float bump3 = TexSample(material.bump, uv + offset[3], false).x;

        float dz = (bump1 - bump0) / delta;
        float dx = (bump3 - bump2) / delta;
        float dy = 0.5f;

        tex_normal = normalize(make_float3(dx, dy, dz));
    }

    // レイ情報等から、正しい座標、法線(法線)を算出
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 ray_position = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
    {
        position = MultiplyRowMatrix3(position, rt_data->local0, rt_data->local1, rt_data->local2);
        normal = CalcNormal(tex_normal, normal, tanget, binormal);
        normal = MultiplyRowMatrix3(normal, rt_data->normal0, rt_data->normal1, rt_data->normal2);
        face_normal = MultiplyRowMatrix3(face_normal, rt_data->normal0, rt_data->normal1, rt_data->normal2);
        
    }

    PayloadData* prd = MergePointer(optixGetPayload_0(), optixGetPayload_1());

    BRDFMaterial brdf_material = InitBRDFMaterial(material, base_color);
    if( prd->depth == 0 )
    {
        prd->albedo = make_float3(base_color);
        prd->normal = normal;
        prd->position = make_float3(material.emission[0], material.emission[1], material.emission[2]);
    }

    if(brdf_material.specular_trans > 0.0f)
    {
        float gamma = acos(dot(ray_dir, params.direct_light.dir));
        float theta = acos(dot(ray_dir, make_float3(0.0f, 1.0f, 0.0f)));
        bool lower_hemisphere = theta >= 0.5f * M_PIf;
        if(lower_hemisphere)
        {
            theta = M_PIf - theta;
        }
        float3 emitter = EvaluateSky(params.direct_light.sky_state, ray_dir, ray_position, params.direct_light.dir, params.direct_light.sky_intensity);
        prd->done = true;
        prd->radiance += emitter * prd->throughput;
        return;
    }

    if(material.emission[0] > 0.0f
    || material.emission[1] > 0.0f
    || material.emission[2] > 0.0f)
    {
        prd->radiance += prd->throughput * make_float3(material.emission[0], material.emission[1], material.emission[2]);
        prd->done = true;
        return;
    }

    // light
    {
        
        float3 shadow_ray_origin = position + normal * 0.01f;

        // direct light
        {
            float3 shadow_ray_direction = normalize(params.direct_light.dir);
            PayloadData shadow_prd;
            traceShadowRay(params.handle, shadow_ray_origin, shadow_ray_direction, 0.0001f, 1e9f, shadow_prd);
            if(!shadow_prd.occluded)
            {
                BRDFSample brdf_sample;
                brdf_sample.f = EvaluateDisneyBRDF(brdf_material, -ray_dir, normal, shadow_ray_direction);
                brdf_sample.pdf = PdfDisneyBRDF(brdf_material, -ray_dir, normal, shadow_ray_direction);
                float light_pdf = 1.0f;
                float mis_weight = CalculateMisWeight(light_pdf, brdf_sample.pdf);
                float absCosTheta = AbsCosTheta(normal, shadow_ray_direction);
                float3 weight = clamp((prd->throughput * mis_weight * brdf_sample.f * absCosTheta) / light_pdf,0.0f, 1.0f);
                float3 le = params.direct_light.emission;
                prd->radiance += weight * le;
            }
        }

        // sky light(lambert sampling)
        {
            float3 shadow_ray_direction;
            float2 rand_value = random_cmj2(prd->seed.cmj_seed);
            CosineSampleHemisphere(rand_value.x, rand_value.y, shadow_ray_direction);
            Onb onb(normal);
            onb.inverse_transform(shadow_ray_direction);
            PayloadData shadow_prd;
            traceShadowRay(params.handle, shadow_ray_origin, shadow_ray_direction, 0.0001f, 1e9f, shadow_prd);
            if(!shadow_prd.occluded)
            {
                BRDFSample brdf_sample;
                brdf_sample.f = EvaluateDisneyBRDF(brdf_material, -ray_dir, normal, shadow_ray_direction);
                brdf_sample.pdf = PdfDisneyBRDF(brdf_material, -ray_dir, normal, shadow_ray_direction);
                float light_pdf =  AbsCosTheta(normal, shadow_ray_direction) * M_PIf;
                float mis_weight = CalculateMisWeight(light_pdf, brdf_sample.pdf);
                float absCosTheta = AbsCosTheta(normal, shadow_ray_direction);
                float3 weight = clamp((prd->throughput * mis_weight * brdf_sample.f * absCosTheta) / light_pdf,0.0f, 1.0f);
                float3 le = EvaluateSky(params.direct_light.sky_state, shadow_ray_direction, shadow_ray_origin, params.direct_light.dir, params.direct_light.sky_intensity);
                prd->radiance += weight * le;
            }
        }
        if(params.light_count > 0)
        {
            for(int32_t i = 0; i < min(5, params.light_count); i++)
            {
                float rand_value = random_cmj1(prd->seed.cmj_seed) * params.light_count;
                unsigned int light_index = unsigned int(rand_value);
                AnyLight& light = params.lights[light_index];
                if(light.type == SphereLight)
                {
                    LightSample light_sample;
                    SampleLight(light, random_cmj3(prd->seed.cmj_seed), light_sample, shadow_ray_origin);
                    PayloadData shadow_prd;
                    traceShadowRay(params.handle, light_sample.origin, light_sample.dir, 0.0001f, light_sample.distance, shadow_prd);
                    if(!shadow_prd.occluded)
                    {
                        BRDFSample brdf_sample;
                        brdf_sample.f = EvaluateDisneyBRDF(brdf_material, -ray_dir, normal, light_sample.dir);
                        brdf_sample.pdf = PdfDisneyBRDF(brdf_material, -ray_dir, normal, light_sample.dir);
                        float light_pdf = light_sample.pdf;
                        float mis_weight = CalculateMisWeight(light_pdf, brdf_sample.pdf);
                        float absCosTheta = AbsCosTheta(normal, light_sample.dir);
                        float3 weight = clamp((prd->throughput * mis_weight * brdf_sample.f * absCosTheta) / light_pdf,0.0f, 1.0f);
                        float3 le = light_sample.emission;
                        prd->radiance += weight * le;
                    }
                }
            }
        }
    }

    // next ray
    {
        float3 rand_value = random_cmj3(prd->seed.cmj_seed);
        BRDFSample brdf_sample;
        brdf_sample.dir = SampleDisneyBRDF(brdf_material, rand_value, -ray_dir, normal);
        brdf_sample.f = EvaluateDisneyBRDF(brdf_material, -ray_dir, normal, brdf_sample.dir);
        brdf_sample.pdf = PdfDisneyBRDF(brdf_material, -ray_dir, normal, brdf_sample.dir);

        bool is_transmitter = false;
        float3 origin_offset = is_transmitter ? -face_normal : face_normal;
        prd->origin = position + origin_offset * 0.00001f;
        prd->direction = brdf_sample.dir;
        if(brdf_sample.pdf > 0.0f)
        {
            float absCosTheta  =AbsCosTheta(brdf_sample.dir, normal);
            prd->throughput *= brdf_sample.f * absCosTheta / brdf_sample.pdf;
        }
        else
        {
            prd->throughput *= 0.0f;
        }
       
        prd->done = false;
    }
    if(prd->depth == 0)
    {
        prd->position = prd->direction;
    }
}