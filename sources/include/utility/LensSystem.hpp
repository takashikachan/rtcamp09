#pragma once

#include "graphic/CudaPixelBuffer.hpp"

namespace slug
{
    struct LensSystemParam 
    {
        bool enable_lensflare;
        int width = 0;
        int height = 0;
        int blade_count = 6;
        int anamorphic = 1;
        int degree = 3;
        float distance = 5000000;
        float sample_mul = 5;
        float r_entrance = 19.5;
        float defocus = 0.0;
        int num_lambdas = 5;
        int filter_size = 1;
        float exposure = 1.0;
        float lambda_from = 440;
        float lambda_to = 660;
        float sensor_width = 80;
        int sensor_xres = 1920;
        int sensor_yres = 1080;
        std::string system_definition_file = "";
        bool dirty = true;
    };

    struct OutputImage
    {
        uint32_t width = 0;
        uint32_t height = 0;
        float* data;
    };

    struct CudaTerm3
    {
        float coefficient;
        int exponents[3];
    };

    struct CudaPoly3
    {
        int trunc_degree;
        CudaTerm3 terms[30];
        int term_count;
        bool consolidated;
    };

    struct CudaPolySystem33
    {
        CudaPoly3 equations[3];
        int trunc_degree;
    };

    struct CudaLensParam 
    {
        float lambda_from;
        float lambda_to;
        float sensor_width;
        float magnification;
        float r_pupil;
        float pixel_size;
        float anamorphic;
        float sensor_scaling;
        float sensor_xres;
        float sensor_yres;
        float sample_mul;
        float exposure;
        float blade_coumt;
    };

    class LensSystem
    {
    public:
        LensSystem();
        ~LensSystem();
        void SetupParam(LensSystemParam& param);
        void CalcLensSystemImage(std::vector<float4>& src, std::vector<float4>& dst);
        std::vector<CudaPolySystem33>& GetSystem() 
        {
            return m_systems;
        }

        CudaLensParam& GetCudaLensParam() 
        {
            return m_cuda_param;
        }

        std::vector<float>& GetRGBParam();
        std::vector<float>& GetBladePositions();
    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
        std::vector<CudaPolySystem33> m_systems;
        LensSystemParam  m_param;
        CudaLensParam m_cuda_param;
    };
}