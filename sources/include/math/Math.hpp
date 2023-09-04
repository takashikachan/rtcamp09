/**
 * @file    Math.hpp
 * @brief   算術系の定義ファイル
 */

#pragma once

#include "utility/Define.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif
#ifndef M_1_PIf
#define M_1_PIf 0.318309886183790671538f
#endif
namespace slug
{

//------------------------------------------------------------------------------------------
// 基本算術系
//------------------------------------------------------------------------------------------
template<typename T>
inline T TMax(T a, T b)
{
    return a > b ? a : b;
}

template<typename T>
inline T TMin(T a, T b)
{
    return a > b ? b : a;
}

template<typename T>
inline T TClamp(T v, T min_v, T max_v)
{
    return TMin(TMax(v, min_v), max_v);
}

template<typename T>
inline T TSaturate(T v)
{
    return TMin(TMax(v, static_cast<T>(0.0)), static_cast<T>(1.0));
}

inline float Lerp(float v0, float v1,float t)
{
    float tmp_t = TSaturate(t);
    return v0 * (1.0f - tmp_t) + v1 * tmp_t;
}

inline double Lerp(double v0, double v1, double t)
{
    double tmp_t = TSaturate(t);
    return v0 * (1.0f - tmp_t) + v1 * tmp_t;
}

template<typename T>
inline T TAbs(T val)
{
    return val < 0 ? -val : val;
}

inline float Degrees(float radians)
{
    return radians * (180.f / M_PIf);
}

inline float Radians(float degrees)
{
    return degrees * (M_PIf / 180.f);
}

template <typename T>
inline T RoundUp(T x, T y)
{
    return ((x + y - 1) / y) * y;
}

struct Transform {
public:
    Transform()
    {
        world = glm::mat4x4(1.0f);
        normal = glm::mat4x4(1.0f);
    }

    glm::mat4x4 adjoint(glm::mat4x4& m)
    {
        glm::mat4x4 mat = glm::mat4x4(1.0f);
        mat[0][0] = m[1][1] * m[2][2] - m[2][1] * m[1][2];
        mat[1][0] = m[2][1] * m[0][2] - m[0][1] * m[2][2];
        mat[2][0] = m[0][1] * m[1][2] - m[1][1] * m[0][2];

        mat[0][1] = m[1][2] * m[2][0] - m[2][2] * m[1][0];
        mat[1][1] = m[2][2] * m[0][0] - m[0][2] * m[2][0];
        mat[2][1] = m[0][2] * m[1][0] - m[1][2] * m[0][0];

        mat[0][2] = m[1][0] * m[2][1] - m[2][0] * m[1][1];
        mat[1][2] = m[2][0] * m[0][1] - m[0][0] * m[2][1];
        mat[2][2] = m[0][0] * m[1][1] - m[1][0] * m[0][1];

        mat[3][0] = mat[3][1] = mat[3][2] = mat[0][3] = mat[1][3] = mat[2][3] = 0.0f;
        mat[3][3] = 1.0f;

        return mat;
    }

    void update(bool is_matrix = true)
    {
        if (is_matrix) {
            // 移動、回転、拡縮から行列を更新
            world = glm::mat4x4(1.0f);
            normal = glm::mat4x4(1.0f);

            world = glm::translate(world, translate);
            world *= glm::toMat4(rotation);
            world = glm::scale(world, scale);

            normal = world;
            normal = world;
            normal = adjoint(normal);
            normal = glm::transpose(normal);
        }
        else
        {
            // 行列から移動、回転、拡縮を更新
            translate = glm::vec3(world[3][0], world[3][1], world[3][2]);

            glm::vec3 t = world[0];
            glm::vec3 b = world[1];
            glm::vec3 n = world[2];
            scale = glm::vec3((float)t.length(), (float)b.length(), (float)n.length());

            glm::mat3x3 rot_mat;
            rot_mat[0] = glm::normalize(t);
            rot_mat[1] = glm::normalize(b);
            rot_mat[2] = glm::normalize(n);

            rotation = glm::toQuat(rot_mat);
        }
    }

    Transform operator*(const Transform& trs)
    {
        Transform ret = {};
        ret.translate = this->translate + trs.translate;
        ret.rotation = this->rotation * trs.rotation;
        ret.scale = this->scale * trs.scale;
        ret.update();
        return ret;
    }
public:
    glm::vec3 translate = glm::vec3(0.0f);
    glm::quat rotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
    glm::vec3 scale = glm::vec3(1.0f);
    glm::mat4x4 world = glm::mat4x4(1.0f);
    glm::mat4x4 normal = glm::mat4x4(1.0f);
};

} // namespace slug