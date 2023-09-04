/**
* @file    CudaPixelBuffer.hpp
* @brief   Cuda用の出力バッファの定義ファイル
*/

#pragma once

#include <vector>

#include "utility/Define.hpp"

namespace slug
{
template<typename PIXEL_FORMAT>
class CudaPixelBuffer;

using SdrPixelBuffer = CudaPixelBuffer<uchar4>;
using HdrPixelBuffer = CudaPixelBuffer<float4>;

/**
    * @brief バッファの種類
*/
enum class CudaPixelBufferType
{
    CUDA_DEVICE = 0,
    ZERO_COPY,
    CUDA_P2P
};

/**
    * @brief Cuda用の出力バッファクラス
    * @tparam PIXEL_FORMAT フォーマット
*/
template<typename PIXEL_FORMAT>
class CudaPixelBuffer
{
public:
    /**
        * @brief コンストラクタ
    */
    CudaPixelBuffer();

    /**
        * @brief コンストラクタ
        * @param type バッファの種類
        * @param width 横サイズ
        * @param height  縦サイズ
    */
    CudaPixelBuffer(CudaPixelBufferType type, int32_t width, int32_t height);

    /**
        * @brief デストラクタ
    */
    ~CudaPixelBuffer();

    /**
        * @brief 生成処理
        * @param type バッファの種類
        * @param width 横サイズ
        * @param height  縦サイズ
    */
    void Create(CudaPixelBufferType type, int32_t width, int32_t height);

    /**
        * @brief リサイズ処理
        * @param width 横サイズ
        * @param height 縦サイズ
    */
    void Resize(int32_t width, int32_t height);

    /**
        * @brief Map
        * @return Cudaに渡す側のポインタ
    */
    PIXEL_FORMAT* Map();

    /**
        * @brief Unmap
    */
    void Unmap();

    /**
        * @brief デバイスインデックスを設定
    */
    void SetDeviceIndex(int32_t device_idx) { m_device_idx = device_idx; }

    /**
        * @brief CudaStreamを設定
    */
    void SetStream(CUstream stream) { m_stream = stream; }

    /**
        * @brief 横サイズを取得
    */
    int32_t GetWidth() const { return m_width; }

    /**
        * @brief 縦サイズを取得
    */
    int32_t GetHeight() const { return m_height; }

    /**
        * @brief バッファのバイトサイズを取得
    */
    size_t GetByteSize();

    /**
        * @brief CPU側のポインタを取得
        * @return
    */
    PIXEL_FORMAT* GetHostPointer();
private:
    /**
        * @brief Cudaデバイスを設定
    */
    void MakeCurrent();
private:
    CudaPixelBufferType m_type = CudaPixelBufferType::CUDA_DEVICE;  //!< バッファの種類
    cudaGraphicsResource* m_cuda_resource = nullptr;                //!< Cuda用のグラフィックスリソース
    PIXEL_FORMAT* m_device_pixels = nullptr;                        //!< Cudaに渡す側のポインタ
    PIXEL_FORMAT* m_host_zerocopy_pixels = nullptr;                 //!< Zeroクリア用
    std::vector<PIXEL_FORMAT> m_host_pixels = {};                   //!< CPU側で扱うポインタ
    CUstream m_stream = 0u;                                         //!< Cudaストリーム
    int32_t m_width = 0u;                                           //!< 横サイズ
    int32_t m_height = 0u;                                          //!< 縦サイズ
    int32_t m_device_idx = 0u;                                      //!< デバイスのインデックス
    uint32_t m_pbo = 0u;                                            //!< pbo
};

template <typename PIXEL_FORMAT>
CudaPixelBuffer<PIXEL_FORMAT>::CudaPixelBuffer()
{
}

template <typename PIXEL_FORMAT>
CudaPixelBuffer<PIXEL_FORMAT>::CudaPixelBuffer(CudaPixelBufferType type, int32_t width, int32_t height)
    :m_type(type)
    , m_width(width)
    , m_height(height)
{
    Create(type, width, height);
}

template <typename PIXEL_FORMAT>
void CudaPixelBuffer<PIXEL_FORMAT>::Create(CudaPixelBufferType type, int32_t width, int32_t height)
{
    if (width < 1 || height < 1)
    {
        ASSERT("CUDAOutputBuffer dimensions must be at least 1 in both x and y.");
        return;
    }

    Resize(width, height);

}

template <typename PIXEL_FORMAT>
CudaPixelBuffer<PIXEL_FORMAT>::~CudaPixelBuffer()
{
    try
    {
        MakeCurrent();
        if (m_type == CudaPixelBufferType::CUDA_DEVICE || m_type == CudaPixelBufferType::CUDA_P2P)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_pixels)));
        }
        else if (m_type == CudaPixelBufferType::ZERO_COPY)
        {
            CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(m_host_zerocopy_pixels)));
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "CUDAOutputBuffer destructor caught exception: " << e.what() << std::endl;
    }
}

template <typename PIXEL_FORMAT>
void CudaPixelBuffer<PIXEL_FORMAT>::Resize(int32_t width, int32_t height)
{
    if (width < 1 || height < 1)
    {
        ASSERT("CUDAOutputBuffer dimensions must be at least 1 in both x and y.");
        return;
    }

    m_width = width;
    m_height = height;

    MakeCurrent();

    if (m_type == CudaPixelBufferType::CUDA_DEVICE || m_type == CudaPixelBufferType::CUDA_P2P)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_pixels)));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&m_device_pixels),
            m_width * m_height * sizeof(PIXEL_FORMAT)
        ));

    }

    if (m_type == CudaPixelBufferType::ZERO_COPY)
    {
        CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(m_host_zerocopy_pixels)));
        CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&m_host_zerocopy_pixels),
            m_width * m_height * sizeof(PIXEL_FORMAT),
            cudaHostAllocPortable | cudaHostAllocMapped
        ));
        CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&m_device_pixels),
            reinterpret_cast<void*>(m_host_zerocopy_pixels),
            0 /*flags*/
        ));
    }

    m_host_pixels.resize(m_width * m_height);
}

template <typename PIXEL_FORMAT>
PIXEL_FORMAT* CudaPixelBuffer<PIXEL_FORMAT>::Map()
{
    switch (m_type)
    {
    case CudaPixelBufferType::CUDA_DEVICE:
    case CudaPixelBufferType::CUDA_P2P:
    default:
        break;
    }

    return m_device_pixels;
}

template <typename PIXEL_FORMAT>
void CudaPixelBuffer<PIXEL_FORMAT>::Unmap()
{
    MakeCurrent();
    switch (m_type)
    {
    case CudaPixelBufferType::CUDA_DEVICE:
    case CudaPixelBufferType::CUDA_P2P:
        CUDA_CHECK(cudaStreamSynchronize(m_stream));
        break;
    default:
        CUDA_CHECK(cudaStreamSynchronize(m_stream));
        break;
    }
}

template <typename PIXEL_FORMAT>
PIXEL_FORMAT* CudaPixelBuffer<PIXEL_FORMAT>::GetHostPointer()
{
    if (m_type == CudaPixelBufferType::CUDA_DEVICE ||
        m_type == CudaPixelBufferType::CUDA_P2P)
    {
        m_host_pixels.resize(m_width * m_height);

        MakeCurrent();
        CUDA_CHECK(cudaMemcpy(
            static_cast<void*>(m_host_pixels.data()),
            Map(),
            m_width * m_height * sizeof(PIXEL_FORMAT),
            cudaMemcpyDeviceToHost
        ));
        Unmap();

        return m_host_pixels.data();
    }
    else
    {
        return m_host_zerocopy_pixels;
    }
}

template <typename PIXEL_FORMAT>
void CudaPixelBuffer<PIXEL_FORMAT>::MakeCurrent()
{
    CUDA_CHECK(cudaSetDevice(m_device_idx));
}

template <typename PIXEL_FORMAT>
size_t CudaPixelBuffer<typename PIXEL_FORMAT>::GetByteSize()
{
    size_t element_size = 4;
    return m_width * m_height * element_size;
}
} // namespace slug