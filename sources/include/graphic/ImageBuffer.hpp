/**
* @file    ImageBuffer.hpp
* @brief   
*/

#pragma once
#include "utility/Define.hpp"

namespace slug
{
enum class BufferImageFormat : uint8_t
{
    Ubyte4 = 0,
    Float4,
    Float3
};

struct ImageBuffer
{
    BufferImageFormat pixel_format = BufferImageFormat::Ubyte4;
    void* data = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;

    uint32_t GetByteSize()
    {
        uint32_t element_size = 4;
        switch (pixel_format)
        {
        case BufferImageFormat::Ubyte4:
            element_size = 4;
            break;
        case BufferImageFormat::Float4:
            element_size = 16;
            break;
        case BufferImageFormat::Float3:
            element_size = 12;
            break;
        }
        return width * height * element_size;
    }

    void destroy()
    {
        switch (pixel_format)
        {
        case BufferImageFormat::Ubyte4:
            delete[] reinterpret_cast<uint4*>(data);
            break;
        case BufferImageFormat::Float4:
            delete[] reinterpret_cast<float4*>(data);
            break;
        case BufferImageFormat::Float3:
            delete[] reinterpret_cast<float3*>(data);
            break;
        }
        data = nullptr;
        width = 0;
        height = 0;
    }
};

extern bool SaveImageJPG(ImageBuffer& buffer, std::string filepath);
} // namespace slug