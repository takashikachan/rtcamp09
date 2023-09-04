#pragma once

#include "optix.h"
#include "cuda_runtime.h"

#include "utility/FileSystem.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace slug
{
    struct ImageSubresource 
    {
        size_t row_pitch = 0;       //!< 行毎のバイト数
        size_t depth_pitch = 0;     //!< 行数x行毎のバイト数
        ptrdiff_t data_offset = 0;  //!< データのオフセット情報
        size_t data_size = 0;       //!< 合計のバイト数
    };
    using ImageDataLayout = std::vector<std::vector<ImageSubresource>>;

    struct ImageInfo
    {
        std::string filename = {};
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t mip_levels = 0;
        uint32_t depth = 0;
        uint32_t array_size = 0;
        uint32_t original_bits_per_pixel = 0;
        IBlobPtr internal_data = {};
        cudaResourceViewFormat cuda_format = {};
        cudaTextureReadMode cuda_read_mode = {};
        cudaChannelFormatDesc cuda_channel_desc = {};
        ImageDataLayout data_layout = {};
        bool is_cube = false;
        bool is_srgb = false;
        bool is_bgra = false;
        int32_t flag = 0;
    };

extern bool LoadSTBI(const char* filename, ImageInfo& info);
extern bool LoadDDS(const char* filename, ImageInfo& info);
extern bool LoadImageFile(const char* filename, ImageInfo& info);

} // namespace slug