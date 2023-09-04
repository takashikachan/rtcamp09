/**
 * @file    CudaPixelBuffer.hpp
 * @brief   Cuda用の出力バッファの定ソースファイル
 */

#include "graphic/CudaPixelBuffer.hpp"

namespace slug
{
template <>
size_t CudaPixelBuffer<uchar4>::GetByteSize()
{
    size_t element_size = 4;
    return m_width * m_height * element_size;
}

template <>
size_t CudaPixelBuffer<float4>::GetByteSize()
{
    size_t element_size = 4 * sizeof(float);
    return m_width * m_height * element_size;
}

template <>
size_t CudaPixelBuffer<float3>::GetByteSize()
{
    size_t element_size = 3 * sizeof(float);
    return m_width * m_height * element_size;
}
}