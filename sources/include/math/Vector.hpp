/**
 * @file    Math.hpp
 * @brief   Vectorの算術系の定義ファイル
 */

#pragma once

#include "math/Math.hpp"
namespace slug
{

 //------------------------------------------------------------------------------------------
 // float2 代入
 //------------------------------------------------------------------------------------------
inline float2 make_float2(const float a, const float b)
{
    return float2(a, b);
}

inline float2 make_float2(const float a)
{
    return make_float2(a, a);
}

inline float2 make_float2(const int2 a)
{
    return make_float2(static_cast<float>(a.x), static_cast<float>(a.y));
}

inline float2 make_float2(const uint2 a)
{
    return make_float2(static_cast<float>(a.x), static_cast<float>(a.y));
}

inline float2 operator-(const float2& a)
{
    return make_float2(-a.x, -a.y);
}

//------------------------------------------------------------------------------------------
// float2 比較
//------------------------------------------------------------------------------------------
inline float2 f2minf(const float2& a, const float2& b)
{
    return make_float2(TMin(a.x, b.x), TMin(a.y, b.y));
}

inline float2 f2maxf(const float2& a, const float2& b)
{
    return make_float2(TMax(a.x, b.x), TMax(a.y, b.y));
}

//------------------------------------------------------------------------------------------
// flaot2 add
//------------------------------------------------------------------------------------------
inline float2 operator+(const float2& a, const float2& b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

inline float2 operator+(const float2& a, const float b)
{
    return make_float2(a.x + b, a.y + b);
}

inline float2 operator+(const float a, const float2& b)
{
    return make_float2(a + b.x, a + b.y);
}

//------------------------------------------------------------------------------------------
// float2 sub
//------------------------------------------------------------------------------------------
inline float2 operator-(const float2& a, const float2& b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

inline float2 operator-(const float2& a, const float b)
{
    return make_float2(a.x - b, a.y - b);
}

inline float2 operator-(const float a, const float2& b)
{
    return make_float2(a - b.x, a - b.y);
}

//------------------------------------------------------------------------------------------
// float2 mul
//------------------------------------------------------------------------------------------
inline float2 operator*(const float2& a, const float2& b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

inline float2 operator*(const float2& a, const float b)
{
    return make_float2(a.x * b, a.y * b);
}

inline float2 operator*(const float a, const float2& b)
{
    return make_float2(a * b.x, a * b.y);
}

//------------------------------------------------------------------------------------------
// float2 div
//------------------------------------------------------------------------------------------
inline float2 operator/(const float2& a, const float2& b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}

inline float2 operator/(const float2& a, const float b)
{
    return make_float2(a.x / b, a.y / b);
}

inline float2 operator/(const float a, const float2& b)
{
    return make_float2(a / b.x, a / b.y);
}


//------------------------------------------------------------------------------------------
// float2 other
//------------------------------------------------------------------------------------------
inline float2 Lerp(const float2& a, const float2& b, float t)
{
    return make_float2(Lerp(a.x, b.x, t), Lerp(a.y, b.y, t));
}

inline float2 Clamp(const float2& a, float max_v, float min_v)
{
    return make_float2(TClamp(a.x, min_v, max_v), TClamp(a.y, min_v, max_v));
}

inline float Dot(const float2& a, const float2& b)
{
    return a.x * b.x + a.y * b.y;
}

inline float Cross(const float2& a, const float2& b)
{
    return a.x * b.y + a.y * b.x;
}

inline float Length(const float2& a)
{
    return sqrtf(Dot(a, a));
}

inline float LengthSq(const float2& a)
{
    return Dot(a, a);
}

inline float2 Normalize(const float2& a)
{
    float l = Length(a);
    return make_float2(a.x / l, a.y / l);
}

inline float2 Reflect(const float2& i, const float2& n)
{
    return i - 2.0f * n * Dot(n, i);
}

//------------------------------------------------------------------------------------------
// float3 代入
//------------------------------------------------------------------------------------------
inline float3 make_float3(const float a, const float b, const float c)
{
    return float3(a, b, c);
}

inline float3 make_float3(const float a)
{
    return make_float3(a, a, a);
}

inline float3 make_float3(const int3 a)
{
    return make_float3(static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z));
}

inline float3 make_float3(const uint3 a)
{
    return make_float3(static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z));
}

inline float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

//------------------------------------------------------------------------------------------
// float3 比較
//------------------------------------------------------------------------------------------
inline float3 f3minf(const float3& a, const float3& b)
{
    return make_float3(TMin(a.x, b.x), TMin(a.y, b.y), TMin(a.z, b.z));
}

inline float3 f3maxf(const float3& a, const float3& b)
{
    return make_float3(TMax(a.x, b.x), TMax(a.y, b.y), TMax(a.z, b.z));
}

//------------------------------------------------------------------------------------------
// flaot3 add
//------------------------------------------------------------------------------------------
inline float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline float3 operator+(const float3& a, const float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline float3 operator+(const float a, const float3& b)
{
    return make_float3(a + b.x, a + b.y, a + b.z);
}

//------------------------------------------------------------------------------------------
// float3 sub
//------------------------------------------------------------------------------------------
inline float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline float3 operator-(const float3& a, const float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}

inline float3 operator-(const float a, const float3& b)
{
    return make_float3(a - b.x, a - b.y, a - b.z);
}

//------------------------------------------------------------------------------------------
// float3 mul
//------------------------------------------------------------------------------------------
inline float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline float3 operator*(const float3& a, const float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline float3 operator*(const float a, const float3& b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}

//------------------------------------------------------------------------------------------
// float3 div
//------------------------------------------------------------------------------------------
inline float3 operator/(const float3& a, const float3& b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline float3 operator/(const float3& a, const float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline float3 operator/(const float a, const float3& b)
{
    return make_float3(a / b.x, a / b.y, a / b.z);
}

//------------------------------------------------------------------------------------------
// float3 other
//------------------------------------------------------------------------------------------
inline float3 Lerp(const float3& a, const float3& b, float t)
{
    return make_float3(Lerp(a.x, b.x, t), Lerp(a.y, b.y, t), Lerp(a.z, b.z, t));
}

inline float3 Clamp(const float3& a, float max_v, float min_v)
{
    return make_float3(TClamp(a.x, min_v, max_v), TClamp(a.y, min_v, max_v), TClamp(a.z, min_v, max_v));
}

inline float Dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float3 Cross(const float3& a, const float3& b)
{
    float x = a.y * b.z - a.z * b.y;
    float y = a.z * b.x - a.x * b.z;
    float z = a.x * b.y - a.y * b.x;
    return make_float3(x, y, z);
}

inline float Length(const float3& a)
{
    return sqrtf(Dot(a, a));
}

inline float LengthSq(const float3& a)
{
    return Dot(a, a);
}

inline float3 Normalize(const float3& a)
{
    float l = Length(a);
    return make_float3(a.x / l, a.y / l, a.z / l);
}

inline float3 Reflect(const float3& i, const float3& n)
{
    return i - 2.0f * n * Dot(n, i);
}

//------------------------------------------------------------------------------------------
// float4 代入
//------------------------------------------------------------------------------------------
inline float4 make_float4(const float a, const float b, const float c, const float d)
{
    return float4(a, b, c, d);
}

inline float4 make_float4(const float a)
{
    return make_float4(a, a, a, a);
}

inline float4 make_float4(const int4 a)
{
    return make_float4(static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z), static_cast<float>(a.w));
}

inline float4 make_float4(const uint4 a)
{
    return make_float4(static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z), static_cast<float>(a.w));
}

inline float4 operator-(const float4& a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

//------------------------------------------------------------------------------------------
// float4 比較
//------------------------------------------------------------------------------------------
inline float4 f4minf(const float4& a, const float4& b)
{
    return make_float4(TMin(a.x, b.x), TMin(a.y, b.y), TMin(a.z, b.z), TMin(a.w, b.w));
}

inline float4 f4maxf(const float4& a, const float4& b)
{
    return make_float4(TMax(a.x, b.x), TMax(a.y, b.y), TMax(a.z, b.z), TMax(a.w, b.w));
}

//------------------------------------------------------------------------------------------
// flaot4 add
//------------------------------------------------------------------------------------------
inline float4 operator+(const float4& a, const float4& b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline float4 operator+(const float4& a, const float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

inline float4 operator+(const float a, const float4& b)
{
    return make_float4(a + b.x, a + b.y, a + b.z, a + b.w);
}

//------------------------------------------------------------------------------------------
// float4 sub
//------------------------------------------------------------------------------------------
inline float4 operator-(const float4& a, const float4& b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline float4 operator-(const float4& a, const float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}

inline float4 operator-(const float a, const float4& b)
{
    return make_float4(a - b.x, a - b.y, a - b.z, a - b.w);
}

//------------------------------------------------------------------------------------------
// float4 mul
//------------------------------------------------------------------------------------------
inline float4 operator*(const float4& a, const float4& b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline float4 operator*(const float4& a, const float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline float4 operator*(const float a, const float4& b)
{
    return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

//------------------------------------------------------------------------------------------
// float4 div
//------------------------------------------------------------------------------------------
inline float4 operator/(const float4& a, const float4& b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline float4 operator/(const float4& a, const float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

inline float4 operator/(const float a, const float4& b)
{
    return make_float4(a / b.x, a / b.y, a / b.z, a / b.w);
}

//------------------------------------------------------------------------------------------
// float4 other
//------------------------------------------------------------------------------------------
inline float4 Lerp(const float4& a, const float4& b, float t)
{
    return make_float4(Lerp(a.x, b.x, t), Lerp(a.y, b.y, t), Lerp(a.z, b.z, t), Lerp(a.w, b.w, t));
}

inline float4 Clamp(const float4& a, float max_v, float min_v)
{
    return make_float4(TClamp(a.x, min_v, max_v), TClamp(a.y, min_v, max_v), TClamp(a.z, min_v, max_v), TClamp(a.w, min_v, max_v));
}

inline float Dot(const float4& a, const float4& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline float Length(const float4& a)
{
    return sqrtf(Dot(a, a));
}

inline float LengthSq(const float4& a)
{
    return Dot(a, a);
}
} // namespace slug