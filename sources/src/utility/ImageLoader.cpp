/**
 * @file    ImageLoader.cpp
 * @brief   
 */

#include "utility/ImageLoader.hpp"
#include "utility/FileSystem.hpp"
#include <dxgi1_6.h>
#include "DDS.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <cstdint>
#include <functional>

#define SLUG_D3D11_RESOURCE_MISC_TEXTURECUBE 0x4

using namespace DirectX;


namespace slug
{
struct FormatMapping 
{
    cudaResourceViewFormat cuda_format = {};
    cudaTextureReadMode cude_read_mode = {};
    DXGI_FORMAT dxgi_format = {};
    uint32_t bits_per_pixel = 0;
    bool srgb = false;
};

struct ChanncelDescMapping 
{
    cudaResourceViewFormat cuda_format;
    std::function<cudaChannelFormatDesc()> create_desc_func;
};

const FormatMapping c_format_mappings[] = 
{
    { cudaResViewFormatNone,                        cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_UNKNOWN,                0 , false },
    { cudaResViewFormatUnsignedChar1,               cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R8_UINT,                8 , false },
    { cudaResViewFormatSignedChar1,                 cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R8_SINT,                8 , false },
    { cudaResViewFormatUnsignedChar1,               cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R8_UNORM,               8 , false },
    { cudaResViewFormatSignedChar1,                 cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R8_SNORM,               8 , false },
    { cudaResViewFormatUnsignedChar2,               cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R8G8_UINT,              16 , false },
    { cudaResViewFormatSignedChar2,                 cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R8G8_SINT,              16 , false },
    { cudaResViewFormatUnsignedChar2,               cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R8G8_UNORM,             16 , false },
    { cudaResViewFormatSignedChar2,                 cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R8G8_SNORM,             16 , false },
    { cudaResViewFormatUnsignedShort1,              cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R16_UINT,               16 , false },
    { cudaResViewFormatSignedShort1,                cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R16_SINT,               16 , false },
    { cudaResViewFormatUnsignedShort1,              cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R16_UNORM,              16 , false },
    { cudaResViewFormatSignedShort1,                cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R16_SNORM,              16 , false },
    { cudaResViewFormatHalf1,                       cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R16_FLOAT,              16 , false },
    { cudaResViewFormatUnsignedChar4,               cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_UINT,          32 , false },
    { cudaResViewFormatSignedChar4,                 cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_SINT,          32 , false },
    { cudaResViewFormatUnsignedChar4,               cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_UNORM,         32 , false },
    { cudaResViewFormatUnsignedChar4,               cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,    32 , true },
    { cudaResViewFormatSignedChar4,                 cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_SNORM,         32 , false },
    { cudaResViewFormatUnsignedShort2,              cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R16G16_UINT,            32 , false },
    { cudaResViewFormatSignedChar2,                 cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R16G16_SINT,            32 , false },
    { cudaResViewFormatUnsignedChar2,               cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R16G16_UNORM,           32 , false },
    { cudaResViewFormatSignedChar2,                 cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R16G16_SNORM,           32 , false },
    { cudaResViewFormatHalf2,                       cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R16G16_FLOAT,           32 , false },
    { cudaResViewFormatUnsignedInt1,                cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R32_UINT,               32 , false },
    { cudaResViewFormatSignedInt1,                  cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R32_SINT,               32 , false },
    { cudaResViewFormatFloat1,                      cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R32_FLOAT,              32 , false },
    { cudaResViewFormatUnsignedShort4,              cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R16G16B16A16_UINT,      64 , false },
    { cudaResViewFormatSignedShort4,                cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R16G16B16A16_SINT,      64 , false },
    { cudaResViewFormatFloat4,                      cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R16G16B16A16_FLOAT,     64 , false },
    { cudaResViewFormatUnsignedShort4,              cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R16G16B16A16_UNORM,     64 , false },
    { cudaResViewFormatSignedShort4,                cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_R16G16B16A16_SNORM,     64 , false },
    { cudaResViewFormatUnsignedInt2,                cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R32G32_UINT,            64 , false },
    { cudaResViewFormatSignedInt2,                  cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R32G32_SINT,            64 , false },
    { cudaResViewFormatFloat2,                      cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R32G32_FLOAT,           64 , false },
    { cudaResViewFormatUnsignedInt4,                cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_UINT,      128 , false },
    { cudaResViewFormatSignedInt1,                  cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_SINT,      128 , false },
    { cudaResViewFormatFloat4,                      cudaReadModeElementType,        DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT,     128 , false },
    { cudaResViewFormatUnsignedBlockCompressed1,    cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC1_UNORM,              4 , false },
    { cudaResViewFormatUnsignedBlockCompressed1,    cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC1_UNORM_SRGB,         4 , true },
    { cudaResViewFormatUnsignedBlockCompressed2,    cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC2_UNORM,              8 , false },
    { cudaResViewFormatUnsignedBlockCompressed2,    cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC2_UNORM_SRGB,         8 , true },
    { cudaResViewFormatUnsignedBlockCompressed3,    cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC3_UNORM,              8 , false },
    { cudaResViewFormatUnsignedBlockCompressed3,    cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC3_UNORM_SRGB,         8 , true },
    { cudaResViewFormatUnsignedBlockCompressed4,    cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC4_UNORM,              4 , false },
    { cudaResViewFormatUnsignedBlockCompressed4,    cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC4_SNORM,              4 , false },
    { cudaResViewFormatUnsignedBlockCompressed5,    cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC5_UNORM,              8 , false },
    { cudaResViewFormatUnsignedBlockCompressed5,    cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC5_SNORM,              8 , false },
    { cudaResViewFormatUnsignedBlockCompressed6H,   cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC6H_UF16,              8 , false },
    { cudaResViewFormatUnsignedBlockCompressed6H,   cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC6H_SF16,              8 , false },
    { cudaResViewFormatUnsignedBlockCompressed7,    cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC7_UNORM,              8 , false },
    { cudaResViewFormatUnsignedBlockCompressed7,    cudaReadModeNormalizedFloat,    DXGI_FORMAT::DXGI_FORMAT_BC7_UNORM_SRGB,         8 , true },
};

const ChanncelDescMapping c_channel_desc_mappings[] =
{
    {cudaResViewFormatUnsignedChar1,    [](){ return cudaCreateChannelDesc<uchar1>(); }},
    {cudaResViewFormatUnsignedChar2,    [](){ return cudaCreateChannelDesc<uchar2>(); }},
    {cudaResViewFormatUnsignedChar4,    [](){ return cudaCreateChannelDesc<uchar4>(); }},
    {cudaResViewFormatSignedChar1,      [](){ return cudaCreateChannelDesc<char1>(); }},
    {cudaResViewFormatSignedChar2,      [](){ return cudaCreateChannelDesc<char2>(); }},
    {cudaResViewFormatSignedChar4,      [](){ return cudaCreateChannelDesc<char4>(); }},
    {cudaResViewFormatUnsignedShort1,   [](){ return cudaCreateChannelDesc<ushort1>(); }},
    {cudaResViewFormatUnsignedShort2,   [](){ return cudaCreateChannelDesc<ushort2>(); }},
    {cudaResViewFormatUnsignedShort4,   [](){ return cudaCreateChannelDesc<ushort4>(); }},
    {cudaResViewFormatSignedShort1,     [](){ return cudaCreateChannelDesc<short1>(); }},
    {cudaResViewFormatSignedShort2,     [](){ return cudaCreateChannelDesc<short2>(); }},
    {cudaResViewFormatSignedShort4,     [](){ return cudaCreateChannelDesc<short4>(); }},
    {cudaResViewFormatUnsignedInt1,     [](){ return cudaCreateChannelDesc<uint1>(); }},
    {cudaResViewFormatUnsignedInt2,     [](){ return cudaCreateChannelDesc<uint2>(); }},
    {cudaResViewFormatUnsignedInt4,     [](){ return cudaCreateChannelDesc<uint4>(); }},
    {cudaResViewFormatSignedInt1,       [](){ return cudaCreateChannelDesc<int1>(); }},
    {cudaResViewFormatSignedInt2,       [](){ return cudaCreateChannelDesc<int2>(); }},
    {cudaResViewFormatSignedInt4,       [](){ return cudaCreateChannelDesc<int4>(); }},
    {cudaResViewFormatHalf1,            [](){ return cudaCreateChannelDesc<ushort1>(); }},
    {cudaResViewFormatHalf2,            [](){ return cudaCreateChannelDesc<ushort2>(); }},
    {cudaResViewFormatHalf4,            [](){ return cudaCreateChannelDesc<ushort4>(); }},
    {cudaResViewFormatFloat1,           [](){ return cudaCreateChannelDesc<float1>(); }},
    {cudaResViewFormatFloat2,           [](){ return cudaCreateChannelDesc<float2>(); }},
    {cudaResViewFormatFloat4,           [](){ return cudaCreateChannelDesc<float4>(); }},
};

cudaChannelFormatDesc CreateChannnelDescFromCudaFormat(cudaResourceViewFormat format)
{
    for (auto itr : c_channel_desc_mappings) 
    {
        if (itr.cuda_format == format) 
        {
            return itr.create_desc_func();
        }
    }
    return cudaCreateChannelDesc<float4>();
}

uint32_t BitsPerPixel(cudaResourceViewFormat format)
{
    for (auto& itr : c_format_mappings)
    {
        if (itr.cuda_format == format)
        {
            return itr.bits_per_pixel;
        }
    }
    ASSERT(false);
    return 0;
}

void GetSurfaceInfo(size_t width,
    size_t height,
    cudaResourceViewFormat fmt,
    uint32_t bitsPerPixel,
    size_t* outNumBytes,
    size_t* outRowBytes,
    size_t* outNumRows)
{
    size_t numBytes = 0;
    size_t rowBytes = 0;
    size_t numRows = 0;

    // bc圧縮かどうか、圧縮された4x4ブロックのバイト数を取得
    bool bc = false;
    size_t bpe = 0;
    {
        switch (fmt) {
        case cudaResViewFormatUnsignedBlockCompressed1:
        case cudaResViewFormatUnsignedBlockCompressed4:
            bc = true;
            bpe = 8;
            break;

        case cudaResViewFormatUnsignedBlockCompressed2:
        case cudaResViewFormatUnsignedBlockCompressed3:
        case cudaResViewFormatUnsignedBlockCompressed5:
        case cudaResViewFormatUnsignedBlockCompressed6H:
        case cudaResViewFormatUnsignedBlockCompressed7:
            bc = true;
            bpe = 16;
            break;
        default:
            break;
        }
    }

    // bc圧縮の場合、圧縮されたピクセルブロックをもとにバイト数を計算する。
    if (bc) {
        // ピクセルブロックを計算
        // ピクセルブロックは4x4なので、4を除算
        // 3加算しているのは小数点切り上げ処理の代わり
        size_t numBlocksWide = 0;
        if (width > 0) {
            numBlocksWide = std::max<size_t>(1, (width + 3) / 4);
        }
        size_t numBlocksHigh = 0;
        if (height > 0) {
            numBlocksHigh = std::max<size_t>(1, (height + 3) / 4);
        }
        rowBytes = numBlocksWide * bpe;
        numRows = numBlocksHigh;
        numBytes = rowBytes * numBlocksHigh;
    }
    else {
        // 圧縮じゃない場合は、そのまま計算
        // 7を加算しているのは小数点切り上げ処理の代わり
        rowBytes = (width * bitsPerPixel + 7) / 8;
        numRows = height;
        numBytes = rowBytes * height;
    }

    if (outNumBytes) {
        *outNumBytes = numBytes;
    }

    if (outRowBytes) {
        *outRowBytes = rowBytes;
    }

    if (outNumRows) {
        *outNumRows = numRows;
    }
}

size_t FillTextureInfoOffsets(ImageInfo& info, size_t data_size, ptrdiff_t dataOffset)
{
    info.original_bits_per_pixel = BitsPerPixel(info.cuda_format);

    // テクスチャ配列情報、ミップ情報を取得
    info.data_layout.resize(info.array_size);
    for (uint32_t arraySlice = 0; arraySlice < info.array_size; arraySlice++)
    {
        size_t w = info.width;
        size_t h = info.height;
        size_t d = info.depth;

        std::vector<ImageSubresource>& sliceData = info.data_layout[arraySlice];
        sliceData.resize(info.mip_levels);

        for (uint32_t mipLevel = 0; mipLevel < info.mip_levels; mipLevel++)
        {
            size_t NumBytes = 0;
            size_t RowBytes = 0;
            size_t NumRows = 0;
            GetSurfaceInfo(w, h, info.cuda_format, info.original_bits_per_pixel, &NumBytes, &RowBytes, &NumRows);

            ImageSubresource& levelData = sliceData[mipLevel];
            levelData.data_offset = dataOffset;
            levelData.data_size = NumBytes;
            levelData.row_pitch = RowBytes;
            levelData.depth_pitch = RowBytes * NumRows;

            dataOffset += NumBytes * d;

            // オフセット量がサイズを超えた場合終了
            if (data_size > 0 && dataOffset > static_cast<ptrdiff_t>(data_size))
            {
                return 0;
            }

            // mipmapなので、1になるまで2の累乗分減らす
            w = w >> 1;
            h = h >> 1;
            d = d >> 1;
            if (w == 0) w = 1;
            if (h == 0) h = 1;
            if (d == 0) d = 1;
        }
    }

    return dataOffset;
}

bool LoadSTBI(const char* filename, ImageInfo& info)
{
    int32_t width = 0;
    int32_t height = 0;
    int32_t channel = 0;
    uint8_t * data = stbi_load(filename, &width, &height, &channel, info.flag);
    info.width = (uint32_t)width;
    info.height = (uint32_t)height;
    info.mip_levels = 1;
    info.depth = 1;
    info.array_size = 1;
    info.original_bits_per_pixel = 32;
    info.cuda_channel_desc = cudaCreateChannelDesc<uchar4>();
    info.internal_data = std::make_shared<Blob>(data, (size_t)(width * height * 4));
    info.cuda_format = cudaResViewFormatUnsignedChar4;
    info.cuda_read_mode = cudaReadModeNormalizedFloat;
    info.is_cube = false;
    info.filename = filename;
    info.is_srgb = 1;
    FillTextureInfoOffsets(info, info.internal_data->size(), 0);

    return info.internal_data->data() != nullptr;
}

#define SLUG_ISBITMASK( r,g,b,a ) ( ddpf.RBitMask == r && ddpf.GBitMask == g && ddpf.BBitMask == b && ddpf.ABitMask == a )

bool ConvertDDSFormat(const DDS_PIXELFORMAT& ddpf, cudaResourceViewFormat& format, cudaTextureReadMode& read_mode, bool& is_bgra)
{
    if (ddpf.flags & DDS_RGB) 
    {

        switch (ddpf.RGBBitCount) 
        {
        case 32:
            if (SLUG_ISBITMASK(0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000)) 
            {
                //RGBA_UNORM
                format = cudaResViewFormatUnsignedChar4;
                read_mode = cudaReadModeNormalizedFloat;
                is_bgra = false;
                return true;
            }

            if (SLUG_ISBITMASK(0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000)) 
            {
                //BGRA_UNORM
                format = cudaResViewFormatUnsignedChar4;
                read_mode = cudaReadModeNormalizedFloat;
                is_bgra = true;
                return true;
            }

            if (SLUG_ISBITMASK(0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000)) 
            {
                //BGRX_UNORM
                format = cudaResViewFormatUnsignedChar4;
                read_mode = cudaReadModeNormalizedFloat;
                is_bgra = true;
                return true;
            }

            if (SLUG_ISBITMASK(0x3ff00000, 0x000ffc00, 0x000003ff, 0xc0000000)) 
            {
                // RGB10A2_UNORM
                return false;
            }

            if (SLUG_ISBITMASK(0x0000ffff, 0xffff0000, 0x00000000, 0x00000000)) 
            {
                // RG16_UNORM
                format = cudaResViewFormatUnsignedShort2;
                read_mode = cudaReadModeNormalizedFloat;
                return true;
            }

            if (SLUG_ISBITMASK(0xffffffff, 0x00000000, 0x00000000, 0x00000000)) 
            {
                // R32_FLOAT
                format = cudaResViewFormatFloat1;
                read_mode = cudaReadModeElementType;
                return true;
            }
            break;

        case 24:
            break;

        case 16:
            if (SLUG_ISBITMASK(0x7c00, 0x03e0, 0x001f, 0x8000)) 
            {
                //BGR5A1_UNORM
                return false;
            }

            if (SLUG_ISBITMASK(0xf800, 0x07e0, 0x001f, 0x0000)) 
            {
                //B5G6R5_UNORM
                return false;
            }

            if (SLUG_ISBITMASK(0x0f00, 0x00f0, 0x000f, 0xf000)) 
            {
                //BGRA4_UNORM
                return false;
            }

            break;
        }
    }
    else if (ddpf.flags & DDS_LUMINANCE) 
    {
        if (8 == ddpf.RGBBitCount) 
        {
            if (SLUG_ISBITMASK(0x000000ff, 0x00000000, 0x00000000, 0x00000000)) 
            {
                //R8_UNORM
                format = cudaResViewFormatUnsignedChar1;
                read_mode = cudaReadModeNormalizedFloat;
                return true;
            }

            if (SLUG_ISBITMASK(0x000000ff, 0x00000000, 0x00000000, 0x0000ff00)) 
            {
                //RG8_UNORM
                format = cudaResViewFormatUnsignedChar2;
                read_mode = cudaReadModeNormalizedFloat;
                return true;
            }
        }

        if (16 == ddpf.RGBBitCount) {
            if (SLUG_ISBITMASK(0x0000ffff, 0x00000000, 0x00000000, 0x00000000)) 
            {
                //R16_UNORM
                format = cudaResViewFormatUnsignedShort1;
                read_mode = cudaReadModeNormalizedFloat;
                return true;
            }
            if (SLUG_ISBITMASK(0x000000ff, 0x00000000, 0x00000000, 0x0000ff00)) 
            {
                //RG8_UNORM
                format = cudaResViewFormatUnsignedChar2;
                read_mode = cudaReadModeNormalizedFloat;
                return true;
            }
        }
    }
    else if (ddpf.flags & DDS_ALPHA) {
        if (8 == ddpf.RGBBitCount) 
        {
            //RG8_UNORM
            format = cudaResViewFormatUnsignedChar1;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        }
    }
    else if (ddpf.flags & DDS_BUMPDUDV) {
        if (16 == ddpf.RGBBitCount) {
            if (SLUG_ISBITMASK(0x00ff, 0xff00, 0x0000, 0x0000)) 
            {
                //RG8_SNORM
                format = cudaResViewFormatSignedChar2;
                read_mode = cudaReadModeNormalizedFloat;
                return true;
            }
        }

        if (32 == ddpf.RGBBitCount) {
            if (SLUG_ISBITMASK(0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000)) 
            {
                //RGBA8_SNORM
                format = cudaResViewFormatSignedChar4;
                read_mode = cudaReadModeNormalizedFloat;
                return true;
            }
            if (SLUG_ISBITMASK(0x0000ffff, 0xffff0000, 0x00000000, 0x00000000)) 
            {
                //RG16_SNORM
                format = cudaResViewFormatSignedChar2;
                read_mode = cudaReadModeNormalizedFloat;
                return true;
            }
        }
    }
    else if (ddpf.flags & DDS_FOURCC) {
        if (MAKEFOURCC('D', 'X', 'T', '1') == ddpf.fourCC) 
        {
            format = cudaResViewFormatUnsignedBlockCompressed1;
            read_mode = cudaReadModeNormalizedFloat;
            return true;

        }
        if (MAKEFOURCC('D', 'X', 'T', '3') == ddpf.fourCC) 
        {
            //BC1_UNORM
            format = cudaResViewFormatUnsignedBlockCompressed2;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        }
        if (MAKEFOURCC('D', 'X', 'T', '5') == ddpf.fourCC) 
        {
            //BC2_UNORM
            format = cudaResViewFormatUnsignedBlockCompressed3;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        }

        if (MAKEFOURCC('D', 'X', 'T', '2') == ddpf.fourCC) 
        {
            //BC3_UNORM
            format = cudaResViewFormatUnsignedBlockCompressed2;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        }
        if (MAKEFOURCC('D', 'X', 'T', '4') == ddpf.fourCC) 
        {
            //BC3_UNORM
            format = cudaResViewFormatUnsignedBlockCompressed3;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        }

        if (MAKEFOURCC('A', 'T', 'I', '1') == ddpf.fourCC) 
       {
            //BC4_UNORM
            format = cudaResViewFormatUnsignedBlockCompressed4;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        }
        if (MAKEFOURCC('B', 'C', '4', 'U') == ddpf.fourCC) 
        {
            //BC4_SNORM
            format = cudaResViewFormatUnsignedBlockCompressed4;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        }
        if (MAKEFOURCC('B', 'C', '4', 'S') == ddpf.fourCC) 
        {
            format = cudaResViewFormatUnsignedBlockCompressed4;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        }

        if (MAKEFOURCC('A', 'T', 'I', '2') == ddpf.fourCC) 
        {
            //BC5_UNORM
            format = cudaResViewFormatUnsignedBlockCompressed5;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        }
        if (MAKEFOURCC('B', 'C', '5', 'U') == ddpf.fourCC) 
        {
            //BC5_UNORM
            format = cudaResViewFormatUnsignedBlockCompressed5;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        }
        if (MAKEFOURCC('B', 'C', '5', 'S') == ddpf.fourCC) 
        {
            //BC5_SNORM
            format = cudaResViewFormatUnsignedBlockCompressed5;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        }

        switch (ddpf.fourCC) {
        case 36:
            //RGBA16_UNORM
            format = cudaResViewFormatUnsignedShort2;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        case 110:
            //RGBA16_SNORM
            format = cudaResViewFormatSignedShort2;
            read_mode = cudaReadModeNormalizedFloat;
            return true;
        case 111:
            //R16_FLOAT
            format = cudaResViewFormatHalf1;
            read_mode = cudaReadModeElementType;
            return true;
        case 112:
            //RG16_FLOAT
            format = cudaResViewFormatHalf2;
            read_mode = cudaReadModeElementType;
            return true;
        case 113:
            //RGBA16_FLOAT
            format = cudaResViewFormatHalf4;
            read_mode = cudaReadModeElementType;
            return true;
        case 114:
            //R32_FLOAT
            format = cudaResViewFormatFloat1;
            read_mode = cudaReadModeElementType;
            return true;
        case 115:
            //RG32_FLOAT
            format = cudaResViewFormatFloat2;
            read_mode = cudaReadModeElementType;
            return true;
        case 116:
            //RGBA32_FLOAT
            format = cudaResViewFormatFloat4;
            read_mode = cudaReadModeElementType;
            return true;
        }
    }

    format = cudaResViewFormatNone;
    read_mode = cudaReadModeElementType;
    return true;
}

bool LoadDDS(const char* filename, ImageInfo& info)
{
    DefaultFileSystem fs = {};
    info.internal_data = fs.ReadFile(std::string(filename));
    if (!info.internal_data || !info.internal_data->data() || info.internal_data->size() <= 0)
    {
        return false;
    }

    const DirectX::DDS_HEADER* header = reinterpret_cast<const DirectX::DDS_HEADER*>(static_cast<const char*>(info.internal_data->data()) + sizeof(uint32_t));
    if (header == nullptr) {
        return false;
    }

    // サイズ確認
    if (header->size != sizeof(DirectX::DDS_HEADER) || header->ddspf.size != sizeof(DirectX::DDS_PIXELFORMAT)) {
        return false;
    }

    // DirectXTex10のヘッダか確認
    bool is_dxt10_header = false;
    if (header->ddspf.flags & DDS_FOURCC && MAKEFOURCC('D', 'X', '1', '0') == header->ddspf.fourCC) {
        if (info.internal_data->size() < (sizeof(DirectX::DDS_HEADER) + sizeof(uint32_t) + sizeof(DirectX::DDS_HEADER_DXT10)))
        {
            return false;
        }
        is_dxt10_header = true;
    }

    ptrdiff_t data_offset = sizeof(uint32_t) + sizeof(DirectX::DDS_HEADER) + (is_dxt10_header ? sizeof(DirectX::DDS_HEADER_DXT10) : 0);

    // 基本的な情報を取得
    info.width = header->width;
    info.height = header->height;
    info.mip_levels = header->mipMapCount ? header->mipMapCount : 1;
    info.depth = 1;
    info.array_size = 1;
    
    if (is_dxt10_header)
    {
        const DirectX::DDS_HEADER_DXT10* d3d10ext = reinterpret_cast<const DirectX::DDS_HEADER_DXT10*>(reinterpret_cast<const char*>(header) + sizeof(DirectX::DDS_HEADER));

        // 配列サイズが0(テクスチャが存在しない)場合失敗
        if (d3d10ext->arraySize == 0) 
        {
            return false;
        }

        // フォーマット関連付け情報からフォーマットを取得
        for (const FormatMapping& mapping : c_format_mappings) 
        {
            if (mapping.dxgi_format == d3d10ext->dxgiFormat) 
            {
                info.cuda_format = mapping.cuda_format;
                info.cuda_read_mode = mapping.cude_read_mode;
                info.is_srgb = mapping.srgb;
                break;
            }
        }

        // フォーマットが不明な場合失敗
        if (info.cuda_format == cudaResViewFormatNone) 
        {
            return false;
        }

        // テクスチャの次元数に合わせて、情報を設定
        if (d3d10ext->resourceDimension == DDS_DIMENSION_TEXTURE1D) 
        {
            if ((header->flags & DDS_HEIGHT) && info.height != 1) 
            {
                return false;
            }

            info.height = 1;
            info.array_size = d3d10ext->arraySize;
        }
        else if (d3d10ext->resourceDimension == DDS_DIMENSION_TEXTURE2D) {
            if (d3d10ext->miscFlag & SLUG_D3D11_RESOURCE_MISC_TEXTURECUBE)
            {
                // キューブテクスチャの時、6面用意
                info.array_size = d3d10ext->arraySize * 6;
                info.is_cube = true;
            }
            else 
            {
                info.array_size = d3d10ext->arraySize;
            }
        }
        else if (d3d10ext->resourceDimension == DDS_DIMENSION_TEXTURE3D) {
            // ボリュームテクスチャの場合
            if (!(header->flags & DDS_HEADER_FLAGS_VOLUME)) 
            {
                return false;
            }
            info.depth = header->depth;
        }
        else 
        {
            return false;
        }
    }
    else 
    {
        if (!ConvertDDSFormat(header->ddspf, info.cuda_format, info.cuda_read_mode, info.is_bgra)) 
        {
            return false;
        }

        if (info.cuda_format == cudaResViewFormatNone)
        {
            return false;
        }

        if (header->flags & DDS_HEADER_FLAGS_VOLUME) 
        {
            // @todo
            info.depth = header->depth;
        }
        else 
        {
            if (header->caps2 & DDS_CUBEMAP)
            {
                if ((header->caps2 & DDS_CUBEMAP_ALLFACES) != DDS_CUBEMAP_ALLFACES) 
                {
                    return false;
                }

                info.array_size = 6;
                info.is_cube = true;
            }
            else 
            {
                info.is_cube = false;
            }
        }
    }

    if (FillTextureInfoOffsets(info, info.internal_data->size(), data_offset) == 0) 
    {
        return false;
    }
    info.filename = filename;
    info.cuda_channel_desc = CreateChannnelDescFromCudaFormat(info.cuda_format);
    return true;
}

bool LoadImageFile(const char* filename, ImageInfo& info) 
{
    std::string filename_str = filename;
    std::string ext_str = filename_str.substr(filename_str.find("."), filename_str.size());
    if (ext_str == ".jpg" || ext_str == ".png")
    {
        info.flag = 4;
        return LoadSTBI(filename, info);
    }
    else if (ext_str == ".dds")
    {
        return LoadDDS(filename, info);
    }
    return false;
}

} // namespace slug