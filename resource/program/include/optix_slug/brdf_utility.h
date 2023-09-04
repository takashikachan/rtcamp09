#pragma once

#include "optix_slug/utility.h"

struct BRDFMaterial
{
    float3 base_color;
    float ior;
    float relative_ior;
    float specular_tint;
    float specular_trans;
    float sheen_tint;
    float roughness;
    float metallic;
    float clearcoat;
    float clearcoat_gloss;
    float sheen;
    float subsurface;
    float anisotropic;

    float debug_specular;
    float debug_diffuse;
};

struct BRDFSample
{
    float3 dir;
    float3 f;
    float pdf;
};

static __host__ __device__ __inline__ float SchlickFresnel(float u) {
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2 * m;
}

static __host__ __device__ __inline__ float GTR1(float NDotH, float a) {
    if (a >= 1.0)
        return (1.0 / M_PIf);
    float a2 = a * a;
    float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
    return (a2 - 1.0) / (M_PIf * log(a2) * t);
}

static __host__ __device__ __inline__ float GTR2(float NDotH, float a) {
    float a2 = a * a;
    float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
    return a2 / (M_PIf * t * t);
}

static __host__ __device__ __inline__ float GTR2_aniso(float NDotH, float HDotX, float HDotY, float ax, float ay) {
    float a = HDotX / ax;
    float b = HDotY / ay;
    float c = a * a + b * b + NDotH * NDotH;
    return 1.0 / (M_PIf * ax * ay * c * c);
}

static __host__ __device__ __inline__ float SmithG_GGX_aniso(float NDotV, float VDotX, float VDotY, float ax, float ay) {
    float a = VDotX * ax;
    float b = VDotY * ay;
    float c = NDotV;
    return 1.0 / (NDotV + sqrt(a * a + b * b + c * c));
}

static __host__ __device__ __inline__ float SmithG_GGX(float NDotV, float alphaG) {
    float a = alphaG * alphaG;
    float b = NDotV * NDotV;
    return 1.0 / (NDotV + sqrt(a + b - a * b));
}

static __host__ __device__ __inline__ float PdfDisneyBRDF(BRDFMaterial& material, float3 V, float3 N, float3 L)
{

    Onb onb(N);
    float3 T = onb.m_tangent;
    float3 B = onb.m_binormal;
    float3 H = normalize(L + V);

    float brdfPdf = 0.0;

    float NDotH = abs(dot(N, H));

    if (dot(N, L) <= 0.0)
    {
        return 1.0;
    }

    float specularAlpha = max(0.001f, material.roughness);
    float clearcoatAlpha = lerp(0.1f, 0.001f, material.clearcoat_gloss);

    float diffuseRatio = 0.5f * (1.0f - material.metallic);
    float specularRatio = 1.0f - diffuseRatio;

    float aspect = sqrt(1.0f - material.anisotropic * 0.9f);
    float ax = max(0.001f, material.roughness / aspect);
    float ay = max(0.001f, material.roughness * aspect);

    // PDFs for brdf
    float pdfGTR2_aniso = GTR2_aniso(NDotH, dot(H, T), dot(H, B), ax, ay) * NDotH;
    float pdfGTR1 = GTR1(NDotH, clearcoatAlpha) * NDotH;
    float ratio = 1.0 / (1.0 + material.clearcoat);
    float pdfSpec = lerp(pdfGTR1, pdfGTR2_aniso, ratio) / (4.0f * abs(dot(L, H)));
    float pdfDiff = abs(dot(L, N)) * (1.0f / M_PIf);

    brdfPdf = diffuseRatio * pdfDiff + specularRatio * pdfSpec;

    return brdfPdf;
}


static __host__ __device__ __inline__ float3 SampleDisneyBRDF(BRDFMaterial& material, float3 rand_value, float3 V, float3 N) {

    Onb onb(N);


    float r1 = rand_value.x;
    float r2 = rand_value.y;
    float r3 = rand_value.z;
    
    float3 dir;

    float diffuseRatio = 0.5 * (1.0 - material.metallic);

    if (r3 < diffuseRatio)
    {
        float3 H;
        CosineSampleHemisphere(r1, r2, H);
        onb.inverse_transform(H);
        dir = H;
    }
    else
    {
        float3 H;
        ImportanceSampleGGX(material.roughness, r1, r2, H);
        onb.inverse_transform(H);
        dir = reflect(-1 * V, H);
    }
    return dir;
}

static __host__ __device__ __inline__ float3 EvaluateDisneyBRDF(BRDFMaterial& material, float3 V, float3 N, float3 L) 
{
    Onb onb(N);
    float3 T = onb.m_tangent;
    float3 B = onb.m_binormal;
    float3 H = normalize(L + V);

    float NDotL = abs(dot(N, L));
    float NDotV = abs(dot(N, V));
    float NDotH = abs(dot(N, H));
    float VDotH = abs(dot(V, H));
    float LDotH = abs(dot(L, H));

    float3 brdf = make_float3(0.0f, 0.0f, 0.0f);
    float3 bsdf = make_float3(0.0f, 0.0f, 0.0f);

    if (material.specular_trans < 1.0 && dot(N, L) > 0.0 && dot(N, V) > 0.0)
    {
        float3 Cdlin = material.base_color;
        float Cdlum = 0.3 * Cdlin.x + 0.6 * Cdlin.y + 0.1 * Cdlin.z;

        float3 Ctint = Cdlum > 0.0 ? Cdlin / Cdlum : make_float3(1.0f, 1.0f, 1.0f);
        float3 Cspec0 = lerp(0.08 * lerp(make_float3(1.0, 1.0f, 1.0f), Ctint, material.specular_tint), Cdlin, material.metallic);
        float3 Csheen = lerp(make_float3(1.0, 1.0f, 1.0f), Ctint, material.sheen_tint);

        float FL = SchlickFresnel(NDotL);
        float FV = SchlickFresnel(NDotV);
        float Fd90 = 0.5f + 2.0f * LDotH * LDotH * material.roughness;
        float Fd = lerp(1.0, Fd90, FL) * lerp(1.0f, Fd90, FV);

        float Fss90 = LDotH * LDotH * material.roughness;
        float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
        float ss = 1.25f * (Fss * (1.0f / (NDotL + NDotV) - 0.5f) + 0.5f);

        float aspect = sqrt(1.0 - material.anisotropic * 0.9f);
        float ax = max(0.001, material.roughness / aspect);
        float ay = max(0.001, material.roughness * aspect);
        float Ds = GTR2_aniso(NDotH, dot(H, T), dot(H, B), ax, ay);
        float FH = SchlickFresnel(LDotH);
        float3 Fs = lerp(Cspec0, make_float3(1.0f, 1.0f, 1.0f), FH);
        float Gs = SmithG_GGX_aniso(NDotL, dot(L, T), dot(L, B), ax, ay);
        Gs *= SmithG_GGX_aniso(NDotV, dot(V, T), dot(V, B), ax, ay);

        float3 Fsheen = FH * material.sheen * Csheen;

        float Dr = GTR1(NDotH, lerp(0.1, 0.001, material.clearcoat_gloss));
        float Fr = lerp(0.04, 1.0, FH);
        float Gr = SmithG_GGX(NDotL, 0.25) * SmithG_GGX(NDotV, 0.25);


        brdf = ((1.0 / M_PIf) * lerp(Fd, ss, material.subsurface) * Cdlin + Fsheen) * (1.0 - material.metallic) + Gs * Fs * Ds + 0.25 * material.clearcoat * Gr * Fr * Dr;
    }
    return brdf;
}