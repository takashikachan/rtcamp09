#pragma once

#include "graphic/ObjectData.hpp"
#include "graphic/GpuResouce.hpp"
#include "graphic/GraphicsContext.hpp"

namespace slug 
{
    struct SphereParam 
    {
        std::string name = {};
        float position[3];
        float rotation[4];
        float scale[3];
        float radius;
        CudaMaterial material;
    };

    void GenerateData(data::Scene& object, SphereParam& param, uint32_t material_offset);

    int32_t GenerateSphereLight(GraphicsContext& context, std::string name, CudaLight& light, ResoucePool& resouce_pool);

    void GenerateBSDFSample(GraphicsContext& context, ResoucePool& resouce_pool);

    void GenerateCornelBoxSample(GraphicsContext& context, ResoucePool& resouce_pool);

    struct ParticleSystemParam 
    {
        std::string name = {};
        uint32_t particle_num = 0;
        float radius = {};
        float emission[3] = {};
        float center[3] = {};
        float range;
    };

    class ParticleSystem
    {
    public:
        void GenerarteParticleLight(GraphicsContext& context, ResoucePool& resouce_pool, ParticleSystemParam& param);
        void UpdateParticleLight(GraphicsContext& context, ResoucePool& resouce_pool, CUfunction& cuda_particle, uint32_t framecount);
    private:
        bool initialize = false;
        ParticleSystemParam  m_param;
        std::vector<uint32_t> m_instance_ids = {};
        std::vector<uint32_t> m_light_ids = {};
        std::vector<uint32_t> m_material_ids = {};
        CudaBuffer m_position[2] = {};
        CudaBuffer m_velocity[2] = {};
        CudaBuffer m_scale[2] = {};
        CudaBuffer m_emission[2] = {};
        std::vector<float3> m_host_position = {};
        std::vector<float3> m_host_velocity = {};
        std::vector<float3> m_host_emission = {};
        std::vector<float> m_host_scale = {};
        uint32_t m_buffer_index = 0;
    };
}