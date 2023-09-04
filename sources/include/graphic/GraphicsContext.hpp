/**
 * @file    GraphicsContext.hpp
 * @brief   グラフィックスコンテキストの定義ファイル
 */
#pragma once

#include "utility/FileSystem.hpp"
#include "utility/Camera.hpp"
#include "CudaPixelBuffer.hpp"

#include <cuda_runtime.h>
#include <optix.h>

#include <memory>
#include <array>

namespace slug
{
class GraphicsContext
{
public:
    /**
     * @brief コンストラクタ
    */
    GraphicsContext();

    /**
     * @brief デストラクタ
    */
    virtual ~GraphicsContext();

    /**
     * @brief コンテキストを生成
     * @param enable_debug デバッグ情報を有効にするか
    */
    void CreateContext(bool enable_debug);

    /**
     * @brief バッファを作成
    */
    void CreateBuffer(CUdeviceptr& buffer, size_t size, const void* cpu_data, cudaMemcpyKind memcpy_kind = cudaMemcpyKind::cudaMemcpyHostToDevice);
    
    void UpdateBuffer(CUdeviceptr& buffer, size_t size, const void* cpu_data, cudaMemcpyKind memcpy_kind = cudaMemcpyKind::cudaMemcpyHostToDevice);

    /**
     * @brief インスタンスデータを作成
    */
    void CreateInstance(CUdeviceptr& buffer, void* instances, size_t size, cudaMemcpyKind memcpy_kind = cudaMemcpyKind::cudaMemcpyHostToDevice);

    void UpdateInstance(CUdeviceptr& buffer, void* instances, size_t size, cudaMemcpyKind memcpy_kind = cudaMemcpyKind::cudaMemcpyHostToDevice);

    /**
     * @brief ミップマップなしテクスチャを更新
    */
    void UpdateTextureArray(cudaArray_t& cuda_array, int32_t offset_w, int32_t offset_h, int32_t pitch, int32_t width, int32_t height, cudaMemcpyKind kind, const void* src);

    /**
     * @brief ミップマップなしテクスチャを作成
    */
    void CreateTextureArray(cudaArray_t& cuda_array, int32_t width, int32_t height, cudaChannelFormatDesc& channel_desc, const void* src);

    /**
     * @brief ミップマップなし3Dテクスチャを更新
    */
    void UpdateTextureArray3D(cudaArray_t& cuda_array, int32_t offset_w, int32_t offset_h, int32_t offset_d, int32_t pitch, int32_t width, int32_t height, int32_t depth, cudaMemcpyKind kind,const void* src);

    /**
     * @brief ミップマップなしキューブテクスチャを更新
    */
    void UpdateTextureCube(cudaArray_t& cuda_array, int32_t offset_w, int32_t offset_h, int32_t pitch, int32_t width, int32_t height, cudaMemcpyKind kind, const void* src);

    /**
     * @brief ミップマップなしキューブテクスチャを作成
    */
    void CreateTextureCubeArray(cudaArray_t& cuda_array, int32_t width, int32_t height, cudaChannelFormatDesc& channel_desc, const void* src);

    /**
     * @brief ミップマップなしテクスチャオブジェクト(ビュー)を作成
    */
    void CreateTextureArrayObject(cudaTextureObject_t& cuda_tex, cudaArray_t& cuda_array, const cudaTextureDesc& texture_desc, const cudaResourceViewDesc* view_desc);

    /**
     * @brief 三角形用のASを作成
    */
    void CreateAccelStructureTriangle(OptixBuildInput& build_input, OptixTraversableHandle& traversable_handle, CUdeviceptr& accel_ptr);

    /**
     * @brief インスタンス用のASを作成
    */
    void CreateAccelStructureInstance(OptixBuildInput& build_input, OptixTraversableHandle& traversable_handle, CUdeviceptr& d_temp_buffer, CUdeviceptr& d_comapct_tmp_buffer);

    void UpdateAccelStructureInstance(OptixBuildInput& build_input, OptixTraversableHandle& traversable_handle, CUstream& cu_stream, CUdeviceptr& d_temp_buffer, CUdeviceptr& d_comapct_tmp_buffer);

    /**
     * @brief プログラムモジュールを作成
    */
    void CreateProgramModule(OptixModuleCompileOptions& module_options, OptixPipelineCompileOptions& pipeline_options, IBlobPtr& program_binary, OptixModule* program_module);

    void CreateProgramGroup(OptixModule* program_module, const char* entry_function_name, OptixProgramGroupKind kind, const OptixProgramGroupOptions& group_options, OptixProgramGroup* program_group);
    void CreateProgramHitGroup(OptixModule* program_module, const char* closest_hit_func, const char* any_hit_func, const OptixProgramGroupOptions& group_options, OptixProgramGroup* program_group);
    void CreatePipeline(std::vector<OptixProgramGroup> program_groups, OptixPipelineLinkOptions& pipeline_link_option, OptixPipelineCompileOptions& pipeline_option, OptixPipeline* pipeline);
    void CreateOptixDenoiser(OptixDenoiserOptions& options, OptixDenoiserModelKind model_kind, OptixDenoiser& denoiser);
    
    void AllocateAndMemcpySBT(CUdeviceptr& shader_record, size_t record_size, void* bindingTables);
    void PackRecordHeader(void* program_group, void* bindingTable);
    void Cleanup();
    bool GetEnableDebug();

    void CreateCudaModule(IBlobPtr& program_binary, CUmodule& cuda_module);

    void GetFunctionKernel(CUmodule& cuda_module, std::string function_name, CUfunction& kernel_addr);

    void LauncKernel(CUfunction& kernel_addr, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z, uint32_t shared_mem_byte, CUstream cu_stream, void** kernel_params, void** extra);
private:
    CUdevice m_cuda_device = 0;
    CUcontext m_cuda_context = 0;
    OptixDeviceContext m_context = 0;
    OptixDenoiser m_denoiser = {};
    bool m_enable_debug = false;
};
} // namespace slug