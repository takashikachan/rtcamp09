#pragma once
#include <cstdint>
#include <vector>
#include <math/Vector.hpp>

namespace slug
{
struct AnimPos
{
    float3 pos;
    int key;
};

class Animation 
{
public:
    void AddData(float3 pos);
    bool GetPosition(int key ,float3& out);
private:
    std::vector<AnimPos> m_animpos;
};


}// namespace slug