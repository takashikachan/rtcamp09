#pragma once


static __host__ __device__ __inline__ float CalculateLuminance(float3 value)
{
    return dot(value, make_float3(0.212671f, 0.715160f, 0.072169f));
}

static __host__ __device__ __inline__ float CosTheta(float3 v)
{
    return v.y;
}

static __host__ __device__ __inline__ float SinTheta(float3 v)
{
    float x = CosTheta(v);
    return sqrt(1.0f - (x * x));
}

static __host__ __device__ __inline__ float TanTheta(float3 v)
{
    return SinTheta(v) / CosTheta(v);
}

static __host__ __device__ __inline__ float CosPhi(float3 v)
{
    float sinTheta = SinTheta(v);
    return (sinTheta == 0) ? 1.0f : clamp(v.x / sinTheta, -1.0f, 1.0f);
}

static __host__ __device__ __inline__ float SinPhi(float3 v)
{
    float sinTheta = SinTheta(v);
    return (sinTheta == 0) ? 1.0f : clamp(v.z / sinTheta, -1.0f, 1.0f);
}


static __host__ __device__ __inline__ float Sign(float x)
{
    if(x < 0.0f) {
        return -1.0f;
    }
    else if(x > 0.0f) {
        return 1.0f;
    }

    return 0.0f;
}

static __host__ __device__ __inline__ float lerp(float v0, float v1, float t)
{
    return v0 * (1.0f - t) + v1 * t;
}

static __host__ __device__ __inline__ float smoothstep(float a, float b, float x)
{
  float t = clamp((x - a)/(b - a), 0.0f, 1.0f);
  return t * t * (3.0f - (2.0f * t));
}

static __host__ __device__ __inline__ float3 refract( float3 i, float3 n, float eta )
{
  float cosi = dot(-i, n);
  float cost2 = 1.0f - eta * eta * (1.0f - cosi*cosi);
  float3 t = eta*i + ((eta*cosi - sqrt(abs(cost2))) * n);
  float flag = (float)(cost2 > 0);
  return t * make_float3(flag, flag, flag);
}

static __host__ __device__ __inline__ float3 my_reflect( float3 i, float3 n)
{
    return i - 2.0f * dot(i, n) * n;
}

static __host__ __device__ __inline__ float3 CalcDirecton( uint3 idx, float2 jitter , int width, int height, float3 U, float3 V, float3 W)
{
    float2 d = make_float2(0.0f, 0.0f);
    d.x = ( static_cast<float>( idx.x ) + jitter.x ) / static_cast<float>( width );
    d.y = ( static_cast<float>( idx.y ) + jitter.y ) / static_cast<float>( height );
    d = 2.0f * d - 1;
    return normalize(d.x * U + d.y * V + W);
}

static __forceinline__ __device__ float2 SampleConcentricDisk(const float2 u)
{
  const float2 u0 = 2.0f * u - 1.0f;

  if (u0.x == 0.0f && u0.y == 0.0f) 
  {
    return make_float2(0.0f);
  }

  const float r = abs(u0.x) > abs(u0.y) ? u0.x : u0.y;
  const float theta = abs(u0.x) > abs(u0.y) ? 0.25f * M_PIf * u0.y / u0.x : 0.5f * M_PIf - 0.25f * M_PIf * u0.x / u0.y;
  return make_float2(r * cos(theta), r * sin(theta));
}

static __host__ __device__ __inline__ void CosineSampleHemisphere(const float u1, const float u2, float3& p)
{
#if 0
    const float r = sqrtf( u1 );
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf( phi );
    p.z = r * sinf( phi );
    p.y = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.z*p.z ) );
#elif 0
    const float r = sqrtf( u1 );
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf( phi );
    p.y = r * sinf( phi );
    p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
#else
  const float2 p_disk = SampleConcentricDisk(make_float2(u1, u2));

  p.x = p_disk.x;
  p.z = p_disk.y;
  // Project up to hemisphere.
  p.y = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.z * p.z));
#endif
}

static __host__ __device__ __inline__ void ImportanceSampleGGX(float roughness, float r1, float r2, float3& p) 
{
    float a = max(0.001, roughness);

    float phi = r1 * M_PIf * 2;

    float cosTheta = sqrt((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
    float sinTheta = clamp(sqrt(1.0f - (cosTheta * cosTheta)), 0.0f, 1.0f);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    p = make_float3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
}


static __host__ __device__ __inline__ void CosineSampleSphere(const float u1, const float u2, float3& p) {
    float z = 1 - 2 * u1;
    float r = sqrt(max(0.0f, 1 - z * z));
    float phi = 2 * M_PIf * u2;
    p = make_float3(r * cos(phi), r * sin(phi), z);
}

struct Onb
{
    __host__ __device__ __inline__ Onb(const float3& normal)
    {
        m_normal = normal;

        if( fabs(m_normal.x) > fabs(m_normal.z) )
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y =  m_normal.x;
            m_binormal.z =  0;
        }
        else
        {
            m_binormal.x =  0;
            m_binormal.y = -m_normal.z;
            m_binormal.z =  m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross( m_binormal, m_normal );
    }

    __host__ __device__ __inline__ void inverse_transform(float3& p) const
    {
        p = p.x*m_tangent + p.y*m_normal + p.z*m_binormal;
    }
    __host__ __device__ __inline__ void to_local(float3& p)
    {
        p = make_float3(dot(p, m_tangent), dot(p, m_normal), dot(p, m_binormal));
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

static __host__ __device__ __inline__ float3 CalcTraiangleSample(float3 p0, float3 p1, float3 p2, float2 uv)
{
    return (1.0f - uv.x - uv.y) * p0 + uv.x * p1 + uv.y * p2;
}

static __host__ __device__ __inline__ float2 CalcTraiangleSample(float2 p0, float2 p1, float2 p2, float2 uv)
{
    return (1.0f - uv.x - uv.y) * p0 + uv.x * p1 + uv.y * p2;    
}

static __host__ __device__ __inline__ float CalcTraiangleSample(float p0, float p1, float p2, float2 uv)
{
    return (1.0f - uv.x - uv.y) * p0 + uv.x * p1 + uv.y * p2;    
}

static __host__ __device__ __inline__ float3 MultiplyRowMatrix3(float3 p, float4 mat0, float4 mat1, float4 mat2)
{
    return make_float3(optix_impl::optixMultiplyRowMatrix(make_float4(p, 1.0f), mat0, mat1, mat2));
}

static __host__ __device__ __inline__ float4 TexSample(cudaTextureObject_t texture, float2 uv, bool is_bgra = false)
{
    float4 tex_color = tex2D<float4>(texture, uv.x, uv.y);
    if(is_bgra)
    {
        return make_float4(tex_color.z, tex_color.y, tex_color.x, tex_color.w);
    }
    else
    {
        return make_float4(tex_color.x, tex_color.y, tex_color.z, tex_color.w);        
    }
}

static __host__ __device__ __inline__ float3 CalcNormal(float3 tNormal,float3 BaseNormal,float3 Tangent,float3 Binormal)
{

    float3 normal_v = tNormal.x * Tangent
                    + tNormal.z * Binormal
                    + tNormal.y * BaseNormal;
    //normal_v.y = -normal_v.y;
    return normalize(normal_v);
}

static __host__ __device__ __inline__ float3 ConvertPolarCoordinate(float3 p)
{
    float r = length(p);
    float r_xz = length(make_float2(p.x, p.z));
    float theta = acos(r_xz / r) * lerp(-1.0f, 1.0f, p.y < 0.0f);
    float phai = acos(p.x / r_xz);
    return make_float3(r, theta, phai);
}

static __host__ __device__ __inline__ float2 ConvertPanoramaUV(float3 p)
{
    float3 ruv = ConvertPolarCoordinate(p);
    float2 uv;
    uv.x = ruv.z / (2.0f * M_PIf);
    uv.y = ruv.y / (0.5f * M_PIf);
    uv.y = (uv.y + 1.0f) * 0.5f;
    return uv;
}

static __host__ __device__ __inline__ unsigned int CalcBufferIndex(int width, int height, uint3 launch_index, bool reverse)
{
    if(!reverse)
    {
        return launch_index.x + launch_index.y * width;
    }
    else
    {
        unsigned int index = launch_index.x + launch_index.y * width;
        return (width * height - 1) - index;
    }
}


static __host__ __device__ __inline__ bool Transmit(float3 H, float3 L, float n, float3& V)
{
    float c = dot(L, H);
    if(c < 0.0f) 
    {
        c = -c;
        H = -H;
    }

    float root = 1.0f - n * n * (1.0f - c * c);
    if(root <= 0)
    {
        return false;
    }

    V = (n * c - sqrt(root)) * H - n * L;
    return true;
}


static __host__ __device__ __inline__ float CalculateMisWeight(float pdf0, float pdf1)
{
  return (pdf0) / (pdf0 + pdf1);
}

static __host__ __device__ __inline__ float AbsCosTheta(float3 v0, float3 v1)
{
    return abs(dot(v0,v1));
}