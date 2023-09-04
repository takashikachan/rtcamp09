/**
 * @file    ImageBuffer.cpp
 * @brief
 */


#include "graphic/ImageBuffer.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <vector>

namespace slug
{
struct JPG_RGBA {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;

    JPG_RGBA() : r(0), g(0), b(0), a(0)
    {}

    JPG_RGBA(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a) : r(_r), g(_g), b(_b), a(_a)
    {}

    JPG_RGBA(uint8_t(&_m)[4]) : r(_m[0]), g(_m[1]), b(_m[2]), a(_m[3])
    {}
};
extern bool SaveImageJPG(ImageBuffer& buffer, std::string filepath) 
{
    if (buffer.pixel_format == BufferImageFormat::Ubyte4)
    {
        stbi_write_png(filepath.c_str(), static_cast<int32_t>(buffer.width), static_cast<int32_t>(buffer.height), static_cast<int>(sizeof(JPG_RGBA)), buffer.data, 0);
    }
    else 
    {
        THROW_EXCEPTION("Error Please SaveImage JPG pixel format is Unyte4");
        return false;
    }
    return true;
}
} // namespace slug