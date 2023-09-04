#include "animation/Animation.hpp"

namespace slug
{
void Animation::AddData(float3 pos)
{
    AnimPos animpos = {};
    animpos.pos = pos;
    m_animpos.push_back(animpos);    
}

float3 CatmullRomInterpolation(const float3& p0, const float3& p1, const float3& p2, const float3& p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;

    float3 v0 = (p2 - p0) * 0.5f;
    float3 v1 = (p3 - p1) * 0.5f;

    float h00 = 2 * t3 - 3 * t2 + 1;
    float h01 = -2 * t3 + 3 * t2;
    float h10 = t3 - 2 * t2 + t;
    float h11 = t3 - t2;

    return h00 * p1 + h01 * p2 + h10 * v0 + h11 * v1;
}

bool Animation::GetPosition(int key, float3& out)
{
    float rate = (float)key / 120.0f;
    if (rate >= 0.0f && rate < 0.3f) 
    {
        float tmp_rate = rate / 0.3f;
        out = (1.0f - tmp_rate) * m_animpos.at(0).pos + tmp_rate * m_animpos.at(1).pos;
        return true;
    }
    else if (rate >= 0.3 && rate < 0.6f)
    {
        float tmp_rate = (rate - 0.3f) / 0.3f;
        out = CatmullRomInterpolation(m_animpos.at(0).pos, m_animpos.at(1).pos, m_animpos.at(2).pos, m_animpos.at(3).pos, tmp_rate);
        return true;
    }
    else if (rate >= 0.6 && rate < 1.0f)
    {
        float tmp_rate = (rate - 0.6f) / 0.3f;
        out = (1.0f - tmp_rate) * m_animpos.at(2).pos + tmp_rate * m_animpos.at(3).pos;
        return true;
    }
    return false;
}
}