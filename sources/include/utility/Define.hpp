/**
 * @file    CudaDefine.hpp
 * @brief   Cudaのマクロ定義ファイル
 */

#pragma once

#include <cstdint>
#include <string>
#include <iostream>
#include <sstream>
#include <cassert>

#include <cuda_runtime.h>
#include <optix.h>

#define MAYBE_UNUSED [[maybe_unused]]

#define FALL_THROUGH [[fallthrough]]

#define NO_DISCARD [[nodiscard]]

#define CUDA_SYNC_CHECK() CudaSyncCheck( __FILE__, __LINE__ )

 /**
  * @brief Cudaの同期チェック
 */
inline void CudaSyncCheck(const char* file, unsigned int line)
{
    cudaDeviceSynchronize();
#if 1//defined(MODE_DEBUG)
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA error on synchronize with error '"
            << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
        throw std::runtime_error(ss.str().c_str());
    }
#endif
}

inline int _ConvertSMVer2CoresDRV(int major, int minor)
{
    typedef struct
    {
        int SM;
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
        {0x80,  64},
        {0x86, 128},
        {0x87, 128},
        {0x89, 128},
        {0x90, 128},
        {-1, -1} };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) 
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) 
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }
    return nGpuArchCoresPerSM[0].Cores;
}

#if 1//defined(MODE_DEBUG)

#define THROW_EXCEPTION(txt)                     \
{                                                \
     std::stringstream ss;                       \
     ss << txt;                                  \
     throw std::runtime_error(ss.str().c_str()); \
}                                                \

#define ASSERT(expression)                              \
{                                                       \
     std::stringstream _strstream = {};                 \
    _strstream << __LINE__ << __FILE__;                 \
    bool is_assert = static_cast<bool>(expression);     \
    if(!is_assert) {throw std::runtime_error(_strstream.str().c_str()); } \
}                                                       \

#define ASSERT_MSG(expression, txt)                  \
{                                                    \
    std::stringstream _strstream = {};               \
    _strstream << txt;                               \
    bool is_assert = static_cast<bool>(expression);  \
    if(!is_assert) {throw std::runtime_error(_strstream.str().c_str()); } \
}                                                    \

#define STATIC_ASSERT(expression) static_assert(expression)

#define STATIC_ASSERT_MSG(expression, txt) static_assert(expression, txt)

#define CUDA_CHECK(call) CheckCudaError(call, #call, __FILE__, __LINE__)

#define CUDA_CHECK_ERROR(err) __checkCudaErrors(err, __FILE__, __LINE__)

#define CUDADRV_CHECK(call) \
    do { \
        CUresult error = call; \
        if (error != CUDA_SUCCESS) { \
            std::stringstream ss; \
            const char* errMsg = "failed to get an error message."; \
            cuGetErrorString(error, &errMsg); \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << errMsg \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)

#define OPTIX_CHECK(call) CheckOptixError(call, #call, __FILE__, __LINE__)

#define OPTIX_CHECK_LOG(call)                                                                \
{                                                                                            \
    do                                                                                       \
    {                                                                                        \
        char   LOG[2048];                                                                    \
        size_t LOG_SIZE = sizeof(LOG);                                                       \
        CheckOptixLog(call, LOG, sizeof(LOG), LOG_SIZE, #call, __FILE__, __LINE__);          \
    } while (false);                                                                         \
}                                                                                            \

#define CU_CHECK(call) \
{ \
  const CUresult result = call; \
  if (result != CUDA_SUCCESS) \
  { \
    const char* name; \
    cuGetErrorName(result, &name); \
    std::cerr << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " failed with " << name << " (" << result << ")\n"; \
    ASSERT(!"CU_CHECK fatal"); \
  } \
}

 /**
  * @brief Cudaのエラーチェック(例外を投げます)
  * @param error エラーコード
  * @param call 呼び出し元の関数
  * @param file 呼び出し元のファイル
  * @param line ライン数
 */
inline void CheckCudaError(cudaError_t error, const char* call, const char* file, uint32_t line)
{
    if (error != cudaSuccess)
    {
        std::stringstream ss = {};
        ss << "Cuda Error :" << cudaGetErrorString(error) << ", file : " << file << ", call:" << call << ", line:" << line;
        throw std::runtime_error(ss.str().c_str());
    }
}

/**
 * @brief Optixのエラーチェック(例外を投げます)
 * @param ret リターンコード
 * @param call 呼び出し元の関数
 * @param file 呼び出し元のファイル
 * @param line ライン数
*/
inline void CheckOptixError(OptixResult ret, const char* call, const char* file, uint32_t line)
{
    if (ret != OPTIX_SUCCESS)
    {
        std::stringstream ss = {};
        ss << "Optix Error :" << ret << "file : " << file << ", call:" << call << ", line:" << line;
        throw std::runtime_error(ss.str().c_str());
    }
}

/**
 * @brief Opitxのログ出力
 * @param ret 結果
 * @param log ログ
 * @param sizeof_log ログのサイズ 
 * @param sizeof_log_returned  
 * @param call 読みだし元
 * @param file ファイル
 * @param line ライン
*/
inline void CheckOptixLog(OptixResult ret, const char* log, size_t sizeof_log, size_t sizeof_log_returned, const char* call, const char* file, uint32_t line )
{
    if (ret != OPTIX_SUCCESS)
    {
        std::stringstream ss;
        ss << "Optix Error :  '" << ret << ", call:" << call << ", file : " << file << ",line :" << line
            << ",log :" << log << (sizeof_log_returned > sizeof_log) ? "TRUNCATED" : "";
        throw std::runtime_error(ss.str().c_str());
    }
}

inline void __checkCudaErrors(CUresult err, const char* file, const int line) {
    if (CUDA_SUCCESS != err) {
        const char* errorStr = NULL;
        cuGetErrorString(err, &errorStr);
        std::stringstream ss;
        ss << "checkCudaErrors() Driver API error = " << err << " : " << errorStr << "file : " << file << "line : " << line;
        throw std::runtime_error(ss.str().c_str());
    }
}

#else
#define ASSERT(expression)
#define ASSERT_MSG(expression, txt)
#define STATIC_ASSERT(expression)
#define STATIC_ASSERT_MSG(expression, txt)
#define THROW_EXCEPTION(call) 
#define CUDA_CHECK_ERROR(err)
#define CUDA_CHECK(call) call
#define OPTIX_CHECK(call) call
#define OPTIX_CHECK_LOG(call)                                                                \
{                                                                                            \
    do                                                                                       \
    {                                                                                        \
        char   LOG[2048];                                                                    \
        size_t LOG_SIZE = sizeof(LOG);                                                       \
        call;                                                                                \
    } while (false);                                                                         \
}   
#endif

template <class T>
inline void getCudaAttribute(T* attribute, CUdevice_attribute device_attribute, int device)
{
    CUDA_CHECK_ERROR(cuDeviceGetAttribute(attribute, device_attribute, device));
}