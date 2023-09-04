/**
 * @file    GraphicsContext.hpp
 * @brief   グラフィックスコンテキストの定義ファイル
 */

#include "graphic/GraphicsContext.hpp"
#include "graphic/CudaPixelBuffer.hpp"
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

namespace slug
{
static void ContextLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << level << "][" << tag << "]: " << message << "\n";
}

#if 0
static int GetMaxGflopsDeviceIdDRV() {
    CUdevice current_device = 0;
    CUdevice max_perf_device = 0;
    int device_count = 0;
    int sm_per_multiproc = 0;
    unsigned long long max_compute_perf = 0;
    int major = 0;
    int minor = 0;
    int multiProcessorCount;
    int clockRate;
    int devices_prohibited = 0;

    cuInit(0);
    CUDA_CHECK_ERROR(cuDeviceGetCount(&device_count));

    if (device_count == 0) 
    {
        exit(EXIT_FAILURE);
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count) {
        CUDA_CHECK_ERROR(cuDeviceGetAttribute(
            &multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            current_device));
        CUDA_CHECK_ERROR(cuDeviceGetAttribute(
            &clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, current_device));
        CUDA_CHECK_ERROR(cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, current_device));
        CUDA_CHECK_ERROR(cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, current_device));

        int computeMode;
        getCudaAttribute<int>(&computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, current_device);

        if (computeMode != CU_COMPUTEMODE_PROHIBITED) {
            if (major == 9999 && minor == 9999) {
                sm_per_multiproc = 1;
            }
            else {
                sm_per_multiproc = _ConvertSMVer2CoresDRV(major, minor);
            }

            unsigned long long compute_perf =
                (unsigned long long)(multiProcessorCount * sm_per_multiproc *
                    clockRate);

            if (compute_perf > max_compute_perf) {
                max_compute_perf = compute_perf;
                max_perf_device = current_device;
            }
        }
        else {
            devices_prohibited++;
        }

        ++current_device;
    }

    if (devices_prohibited == device_count) {
        fprintf(stderr,
            "gpuGetMaxGflopsDeviceIdDRV error: all devices have compute mode "
            "prohibited.\n");
        exit(EXIT_FAILURE);
    }

    return max_perf_device;
}
#endif

GraphicsContext::GraphicsContext()
{

}

GraphicsContext::~GraphicsContext()
{

}

void GraphicsContext::CreateContext(bool enable_debug)
{
    m_enable_debug = enable_debug;

    CUDA_CHECK(cudaFree(0));

    CUDA_CHECK_ERROR(cuInit(0));

    OptixDeviceContext context;
    CUcontext cu_ctx = 0;

    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &ContextLogCallback;
    options.logCallbackLevel = 4;

    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));
    m_context = context;
}

void GraphicsContext::CreateBuffer(CUdeviceptr& buffer, size_t size, const void* cpu_data, cudaMemcpyKind memcpy_kind)
{
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffer), size));
    if (cpu_data != nullptr)
    {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(buffer), cpu_data, size, memcpy_kind));
    }
}

void GraphicsContext::UpdateBuffer(CUdeviceptr& buffer, size_t size, const void* cpu_data, cudaMemcpyKind memcpy_kind)
{
    if (cpu_data != nullptr)
    {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(buffer), cpu_data, size, memcpy_kind));
    }
}

void GraphicsContext::CreateInstance(CUdeviceptr& buffer, void* instances, size_t size, cudaMemcpyKind memcpy_kind)
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffer), size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(buffer), instances, size, memcpy_kind));
}

void GraphicsContext::UpdateInstance(CUdeviceptr& buffer, void* instances, size_t size, cudaMemcpyKind memcpy_kind)
{
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(buffer), instances, size, memcpy_kind));
}

void GraphicsContext::UpdateTextureArray(cudaArray_t& cuda_array, int32_t offset_w, int32_t offset_h, int32_t pitch, int32_t width, int32_t height, cudaMemcpyKind kind,const void* src)
{
    CUDA_CHECK(cudaMemcpy2DToArray(cuda_array, offset_w, offset_h, src, pitch, width, height, kind));
}

void GraphicsContext::CreateTextureArray(cudaArray_t& cuda_array, int32_t width, int32_t height, cudaChannelFormatDesc& channel_desc, const void* src)
{
    CUDA_CHECK(cudaMallocArray(&cuda_array, &channel_desc, width, height));
    if (src != nullptr) 
    {
        int32_t byte_size = 0;
        byte_size += channel_desc.x / 8;
        byte_size += channel_desc.y / 8;
        byte_size += channel_desc.z / 8;
        byte_size += channel_desc.w / 8;
        int32_t pitch = width * byte_size;
        UpdateTextureArray(cuda_array, 0, 0, pitch, pitch, height, cudaMemcpyKind::cudaMemcpyHostToDevice, src);
    }
}

void GraphicsContext::UpdateTextureArray3D(cudaArray_t& cuda_array, int32_t offset_w, int32_t offset_h, int32_t offset_d, int32_t pitch, int32_t width, int32_t height, int32_t depth, cudaMemcpyKind kind,const void* src)
{
    cudaMemcpy3DParms params = {};
    params.dstArray = cuda_array;
    params.dstPos = cudaPos{0, 0, 0};

    params.srcPtr = cudaPitchedPtr{ (void*)src, (size_t)pitch, (size_t)width, (size_t)height };
    params.srcPos = cudaPos{ (size_t)offset_w, (size_t)offset_h, (size_t)offset_d };
    params.extent = cudaExtent{ (size_t)width, (size_t)height, (size_t)depth };
    params.kind = kind;
    cudaMemcpy3D(&params);
}

void GraphicsContext::UpdateTextureCube(cudaArray_t& cuda_array, int32_t offset_w, int32_t offset_h, int32_t pitch, int32_t width, int32_t height, cudaMemcpyKind kind,const void* src)
{
    UpdateTextureArray3D(cuda_array, offset_w, offset_h, 0, pitch, width, height, 6, kind, src);
}

void GraphicsContext::CreateTextureCubeArray(cudaArray_t& cuda_array, int32_t width, int32_t height, cudaChannelFormatDesc& channel_desc, const void* src)
{
    cudaExtent ext = { (size_t)width, (size_t)height, 6 };
    cudaMalloc3DArray(&cuda_array, &channel_desc, ext, cudaArrayCubemap);
    if (src != nullptr) 
    {
        int32_t byte_size = 0;
        byte_size += channel_desc.x / 8;
        byte_size += channel_desc.y / 8;
        byte_size += channel_desc.z / 8;
        byte_size += channel_desc.w / 8;
        int32_t pitch = width * byte_size;
        UpdateTextureCube(cuda_array, 0, 0, pitch, width, height, cudaMemcpyKind::cudaMemcpyHostToDevice, src);
    }
}

void GraphicsContext::CreateTextureArrayObject(cudaTextureObject_t& cuda_tex, cudaArray_t& cuda_array, const cudaTextureDesc& texture_desc, const cudaResourceViewDesc* view_desc)
{
    cudaResourceDesc resource_desc = {};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = cuda_array;
    CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &resource_desc, &texture_desc, view_desc));
}

void GraphicsContext::CreateAccelStructureTriangle(OptixBuildInput& build_input, OptixTraversableHandle& traversable_handle, CUdeviceptr& accel_ptr)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accel_options, &build_input, 1, &gas_buffer_sizes));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = RoundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(m_context, 0, &accel_options, &build_input, 1, d_temp_buffer, gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size, gas_buffer_sizes.outputSizeInBytes, &traversable_handle, &emitProperty, 1 ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&accel_ptr), compacted_gas_size));
        OPTIX_CHECK(optixAccelCompact(m_context, 0, traversable_handle, accel_ptr, compacted_gas_size, &traversable_handle));
        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        accel_ptr = d_buffer_temp_output_gas_and_compacted_size;
    }

}

void GraphicsContext::CreateAccelStructureInstance(OptixBuildInput& build_input, OptixTraversableHandle& traversable_handle, CUdeviceptr& d_temp_buffer, CUdeviceptr& d_comapct_tmp_buffer)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accel_options, &build_input, 1, &ias_buffer_sizes));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), ias_buffer_sizes.tempSizeInBytes));

    size_t      compactedSizeOffset = RoundUp<size_t>(ias_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_comapct_tmp_buffer), compactedSizeOffset + 8));

    //OptixAccelEmitDesc emitProperty = {};
    //emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    //emitProperty.result = (CUdeviceptr)((char*)d_comapct_tmp_buffer + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(m_context, 0, &accel_options, &build_input, 1, d_temp_buffer, ias_buffer_sizes.tempSizeInBytes,
        d_comapct_tmp_buffer, ias_buffer_sizes.outputSizeInBytes, &traversable_handle, nullptr, 0));
}

void GraphicsContext::UpdateAccelStructureInstance(OptixBuildInput& build_input, OptixTraversableHandle& traversable_handle, CUstream& cu_stream, CUdeviceptr& d_temp_buffer, CUdeviceptr& d_comapct_tmp_buffer)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accel_options, &build_input, 1, &ias_buffer_sizes));

    if (d_temp_buffer <= 0)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), ias_buffer_sizes.tempSizeInBytes));
    }

    if (d_comapct_tmp_buffer <= 0)
    {
        size_t      compactedSizeOffset = RoundUp<size_t>(ias_buffer_sizes.outputSizeInBytes, 8ull);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_comapct_tmp_buffer), compactedSizeOffset + 8));
    }

    OPTIX_CHECK(optixAccelBuild(m_context, cu_stream, &accel_options, &build_input, 1, d_temp_buffer, ias_buffer_sizes.tempSizeInBytes,
        d_comapct_tmp_buffer, ias_buffer_sizes.outputSizeInBytes, &traversable_handle, nullptr, 0));

    CUresult result = cuStreamSynchronize(cu_stream);
    if (result != CUDA_SUCCESS) 
    {
        std::stringstream ss = {};
        ss << "Cuda Error :" << result;
        throw std::runtime_error(ss.str().c_str());
    }

}

void GraphicsContext::CreateProgramModule(OptixModuleCompileOptions& module_options, OptixPipelineCompileOptions& pipeline_options, IBlobPtr& program_binary, OptixModule* program_module)
{
    if (m_enable_debug)
    {
        module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    }
    else
    {
        module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    }

    if (m_enable_debug)
    {
        pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    }
    else
    {
        pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    }

    OPTIX_CHECK_LOG(optixModuleCreate(m_context, &module_options, &pipeline_options,
        (const char*)program_binary->data(), program_binary->size(), LOG, &LOG_SIZE, program_module));
}


void GraphicsContext::CreateProgramGroup(OptixModule* program_module, const char* entry_function_name, OptixProgramGroupKind kind, const OptixProgramGroupOptions& group_options, OptixProgramGroup* program_group)
{

    OptixProgramGroupDesc prog_group_desc = {};
    prog_group_desc.kind = kind;
    if (kind == OPTIX_PROGRAM_GROUP_KIND_RAYGEN)
    {
        prog_group_desc.raygen.module = *(OptixModule*)program_module;
        prog_group_desc.raygen.entryFunctionName = entry_function_name;
    }
    else if(kind == OPTIX_PROGRAM_GROUP_KIND_MISS)
    {
        prog_group_desc.miss.module = *(OptixModule*)program_module;
        prog_group_desc.miss.entryFunctionName = entry_function_name;
    }
    else if (kind == OPTIX_PROGRAM_GROUP_KIND_EXCEPTION)
    {
        prog_group_desc.exception.module = *(OptixModule*)program_module;
        prog_group_desc.exception.entryFunctionName = entry_function_name;  
    }
    else if (kind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP)
    {
        prog_group_desc.hitgroup.moduleCH = *(OptixModule*)program_module;
        prog_group_desc.hitgroup.entryFunctionNameCH = entry_function_name;
    }
    else if (kind == OPTIX_PROGRAM_GROUP_KIND_CALLABLES)
    {
        prog_group_desc.callables.moduleDC = *(OptixModule*)program_module;
        prog_group_desc.callables.entryFunctionNameDC = entry_function_name;
    }

    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, &prog_group_desc, 1, &group_options, LOG, &LOG_SIZE, program_group));
}

void GraphicsContext::CreateProgramHitGroup(OptixModule* program_module, const char* closest_hit_func, const char* any_hit_func, const OptixProgramGroupOptions& group_options, OptixProgramGroup* program_group)
{
    OptixProgramGroupDesc prog_group_desc = {};
    prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    prog_group_desc.hitgroup.moduleCH = *(OptixModule*)program_module;
    prog_group_desc.hitgroup.entryFunctionNameCH = closest_hit_func;

    prog_group_desc.hitgroup.moduleAH = *(OptixModule*)program_module;
    prog_group_desc.hitgroup.entryFunctionNameAH = any_hit_func;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, &prog_group_desc, 1, &group_options, LOG, &LOG_SIZE, program_group));
}

void GraphicsContext::CreatePipeline(std::vector<OptixProgramGroup> program_groups, OptixPipelineLinkOptions& pipeline_link_option, OptixPipelineCompileOptions& pipeline_option, OptixPipeline* pipeline)
{

    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_context,
        &pipeline_option,
        &pipeline_link_option,
        program_groups.data(),
        static_cast<uint32_t>(program_groups.size()),
        LOG, &LOG_SIZE,
        pipeline
    ));

    OptixStackSizes stack_sizes = {};
    for (size_t i = 0; i < program_groups.size(); i++)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(program_groups.at(i), &stack_sizes, *(OptixPipeline*)pipeline));
    }

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(
        *(OptixPipeline*)pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    ));
}

void GraphicsContext::CreateOptixDenoiser(OptixDenoiserOptions& options, OptixDenoiserModelKind model_kind, OptixDenoiser& denoiser)
{
    OPTIX_CHECK(optixDenoiserCreate(m_context, model_kind, &options, &denoiser));
}

void GraphicsContext::AllocateAndMemcpySBT(CUdeviceptr& shader_record, size_t record_size, void* bindingTables)
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&shader_record), record_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(shader_record), bindingTables, record_size, cudaMemcpyHostToDevice));
}

void GraphicsContext::PackRecordHeader(void* program_group, void* bindingTables)
{
    OPTIX_CHECK(optixSbtRecordPackHeader((OptixProgramGroup)program_group, bindingTables));
}


void GraphicsContext::Cleanup() 
{
    OPTIX_CHECK(optixDeviceContextDestroy(m_context));
}

bool GraphicsContext::GetEnableDebug() 
{
    return m_enable_debug;
}

void GraphicsContext::CreateCudaModule(IBlobPtr& program_binary, CUmodule& cuda_module)
{
    CUDA_CHECK_ERROR(cuModuleLoadData(&cuda_module, reinterpret_cast<const char*>(program_binary->data())));
}

void GraphicsContext::GetFunctionKernel(CUmodule& cuda_module, std::string function_name, CUfunction& kernel_addr)
{
    CUDA_CHECK_ERROR(cuModuleGetFunction(&kernel_addr, cuda_module, function_name.c_str()));
}

void GraphicsContext::LauncKernel(CUfunction& kernel_addr, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z, uint32_t shared_mem_byte, CUstream cu_stream, void** kernel_params, void** extra)
{
    CUDA_CHECK_ERROR(cuLaunchKernel(kernel_addr, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem_byte,cu_stream, kernel_params, extra));
    CUDA_CHECK_ERROR(cuCtxSynchronize());
}
} // namespace slug