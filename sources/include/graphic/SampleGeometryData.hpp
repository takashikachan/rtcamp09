/**
* @file    SampleGeometryData.hpp
* @brief   サンプル描画で使用するジオメトリデータ
*/

#pragma once

#include <array>
#include <vector_types.h>

namespace slug
{
// コーネルボックス
namespace cornel_box
{
/**
    * @brief 頂点データ
*/
struct Vertex
{
    float x, y, z, pad;
};

struct Texcoord
{
    float x, y, z, pad;
};

/**
    * @brief インデックスデータ
*/
struct IndexedTriangle
{
    uint32_t v1, v2, v3, pad;
};

/**
    * @brief インスタンス情報
*/
struct Instance
{
    float transform[12];
};

const int32_t TRIANGLE_COUNT = 32;
const int32_t MAT_COUNT = 4;

// 頂点情報
const static std::array<Vertex, TRIANGLE_COUNT * 3> g_vertices =
{ {
        // Floor  -- white lambert
        {    0.0f,    0.0f,    0.0f, 0.0f},
        {    0.0f,    0.0f,  559.2f, 0.0f},
        {  556.0f,    0.0f,  559.2f, 0.0f},
        {    0.0f,    0.0f,    0.0f, 0.0f},
        {  556.0f,    0.0f,  559.2f, 0.0f},
        {  556.0f,    0.0f,    0.0f, 0.0f},

        // Ceiling -- white lambert
        {    0.0f,  548.8f,    0.0f, 0.0f},
        {  556.0f,  548.8f,    0.0f, 0.0f},
        {  556.0f,  548.8f,  559.2f, 0.0f},

        {    0.0f,  548.8f,    0.0f, 0.0f},
        {  556.0f,  548.8f,  559.2f, 0.0f},
        {    0.0f,  548.8f,  559.2f, 0.0f},

        // Back wall -- white lambert
        {    0.0f,    0.0f,  559.2f, 0.0f},
        {    0.0f,  548.8f,  559.2f, 0.0f},
        {  556.0f,  548.8f,  559.2f, 0.0f},

        {    0.0f,    0.0f,  559.2f, 0.0f},
        {  556.0f,  548.8f,  559.2f, 0.0f},
        {  556.0f,    0.0f,  559.2f, 0.0f},

        // Right wall -- green lambert
        {    0.0f,    0.0f,    0.0f, 0.0f},
        {    0.0f,  548.8f,    0.0f, 0.0f},
        {    0.0f,  548.8f,  559.2f, 0.0f},

        {    0.0f,    0.0f,    0.0f, 0.0f},
        {    0.0f,  548.8f,  559.2f, 0.0f},
        {    0.0f,    0.0f,  559.2f, 0.0f},

        // Left wall -- red lambert
        {  556.0f,    0.0f,    0.0f, 0.0f},
        {  556.0f,    0.0f,  559.2f, 0.0f},
        {  556.0f,  548.8f,  559.2f, 0.0f},

        {  556.0f,    0.0f,    0.0f, 0.0f},
        {  556.0f,  548.8f,  559.2f, 0.0f},
        {  556.0f,  548.8f,    0.0f, 0.0f},

        // Short block -- white lambert
        {  130.0f,  165.0f,   65.0f, 0.0f},
        {   82.0f,  165.0f,  225.0f, 0.0f},
        {  242.0f,  165.0f,  274.0f, 0.0f},

        {  130.0f,  165.0f,   65.0f, 0.0f},
        {  242.0f,  165.0f,  274.0f, 0.0f},
        {  290.0f,  165.0f,  114.0f, 0.0f},

        {  290.0f,    0.0f,  114.0f, 0.0f},
        {  290.0f,  165.0f,  114.0f, 0.0f},
        {  240.0f,  165.0f,  272.0f, 0.0f},

        {  290.0f,    0.0f,  114.0f, 0.0f},
        {  240.0f,  165.0f,  272.0f, 0.0f},
        {  240.0f,    0.0f,  272.0f, 0.0f},

        {  130.0f,    0.0f,   65.0f, 0.0f},
        {  130.0f,  165.0f,   65.0f, 0.0f},
        {  290.0f,  165.0f,  114.0f, 0.0f},

        {  130.0f,    0.0f,   65.0f, 0.0f},
        {  290.0f,  165.0f,  114.0f, 0.0f},
        {  290.0f,    0.0f,  114.0f, 0.0f},

        {   82.0f,    0.0f,  225.0f, 0.0f},
        {   82.0f,  165.0f,  225.0f, 0.0f},
        {  130.0f,  165.0f,   65.0f, 0.0f},

        {   82.0f,    0.0f,  225.0f, 0.0f},
        {  130.0f,  165.0f,   65.0f, 0.0f},
        {  130.0f,    0.0f,   65.0f, 0.0f},

        {  240.0f,    0.0f,  272.0f, 0.0f},
        {  240.0f,  165.0f,  272.0f, 0.0f},
        {   82.0f,  165.0f,  225.0f, 0.0f},

        {  240.0f,    0.0f,  272.0f, 0.0f},
        {   82.0f,  165.0f,  225.0f, 0.0f},
        {   82.0f,    0.0f,  225.0f, 0.0f},

        // Tall block -- white lambert
        {  423.0f,  330.0f,  247.0f, 0.0f},
        {  265.0f,  330.0f,  296.0f, 0.0f},
        {  314.0f,  330.0f,  455.0f, 0.0f},

        {  423.0f,  330.0f,  247.0f, 0.0f},
        {  314.0f,  330.0f,  455.0f, 0.0f},
        {  472.0f,  330.0f,  406.0f, 0.0f},

        {  423.0f,    0.0f,  247.0f, 0.0f},
        {  423.0f,  330.0f,  247.0f, 0.0f},
        {  472.0f,  330.0f,  406.0f, 0.0f},

        {  423.0f,    0.0f,  247.0f, 0.0f},
        {  472.0f,  330.0f,  406.0f, 0.0f},
        {  472.0f,    0.0f,  406.0f, 0.0f},

        {  472.0f,    0.0f,  406.0f, 0.0f},
        {  472.0f,  330.0f,  406.0f, 0.0f},
        {  314.0f,  330.0f,  456.0f, 0.0f},

        {  472.0f,    0.0f,  406.0f, 0.0f},
        {  314.0f,  330.0f,  456.0f, 0.0f},
        {  314.0f,    0.0f,  456.0f, 0.0f},

        {  314.0f,    0.0f,  456.0f, 0.0f},
        {  314.0f,  330.0f,  456.0f, 0.0f},
        {  265.0f,  330.0f,  296.0f, 0.0f},

        {  314.0f,    0.0f,  456.0f, 0.0f},
        {  265.0f,  330.0f,  296.0f, 0.0f},
        {  265.0f,    0.0f,  296.0f, 0.0f},

        {  265.0f,    0.0f,  296.0f, 0.0f},
        {  265.0f,  330.0f,  296.0f, 0.0f},
        {  423.0f,  330.0f,  247.0f, 0.0f},

        {  265.0f,    0.0f,  296.0f, 0.0f},
        {  423.0f,  330.0f,  247.0f, 0.0f},
        {  423.0f,    0.0f,  247.0f, 0.0f},

        // Ceiling light -- emmissive
        {  343.0f,  548.6f,  227.0f, 0.0f},
        {  213.0f,  548.6f,  227.0f, 0.0f},
        {  213.0f,  548.6f,  332.0f, 0.0f},

        {  343.0f,  548.6f,  227.0f, 0.0f},
        {  213.0f,  548.6f,  332.0f, 0.0f},
        {  343.0f,  548.6f,  332.0f, 0.0f}
    } };

//!< マテリアル情報
static std::array<uint32_t, TRIANGLE_COUNT> g_mat_indices = { {
    0, 0,                          // Floor         -- white lambert
    0, 0,                          // Ceiling       -- white lambert
    0, 0,                          // Back wall     -- white lambert
    1, 1,                          // Right wall    -- green lambert
    2, 2,                          // Left wall     -- red lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Short block   -- white lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Tall block    -- white lambert
    3, 3                           // Ceiling light -- emmissive
} };

//!< エミッション情報
const std::array<float3, MAT_COUNT> g_emission_colors =
{ {
    {  0.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f,  0.0f },
    { 15.0f, 15.0f,  5.0f }

} };

// 拡散反射情報
const std::array<float3, MAT_COUNT> g_diffuse_colors =
{ {
    { 0.80f, 0.80f, 0.80f },
    { 0.05f, 0.80f, 0.05f },
    { 0.80f, 0.05f, 0.05f },
    { 0.50f, 0.50f, 0.50f }
} };

}// namespace cornel_box
} // namespace slug