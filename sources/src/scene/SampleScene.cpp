#include "graphic/ObjectData.hpp"
#include "graphic/GpuResouce.hpp"
#include "graphic/GeometryObject.hpp"

#include "scene/SampleScene.hpp"

#include "utility/FileSystem.hpp"
#include "utility/SystemPath.hpp"

#include <optix_stubs.h>
#include <fstream>
#include <filesystem>
#include "math/Vector.hpp"

#include "DirectXTex.h"
#include "utility/LensSystem.hpp"
#include "animation/Animation.hpp"
#include "utility/Timer.hpp"

using namespace std;
#include "optix_slug/default_shader_data.h"
#include "optix_slug/sky_arhosek_utility.h"


namespace slug
{


typedef Record<RayGenData> RayGenRecord;
typedef Record<MissData> MissRecord;
typedef Record<HitGroupData> HitGroupRecord;

/**
 * @brief 内部実装
*/
struct SampleScene::Implement 
{
public:
    ResoucePool resources = {};                         //!< GPUリソースプール

    OptixModule program_module = 0;                     //!< プログラムデータ

    OptixPipelineCompileOptions pipeline_option = {};   //!< パイプラインオプション
    OptixPipeline pipeline = 0;                         //!< パオプライん

    OptixProgramGroup raygen_prog_group = 0;            //!< raygenデータ

    OptixProgramGroup radiance_miss_group = 0;          //!< miss(radiance)データ
    OptixProgramGroup shadow_miss_group = 0;            //!< miss(shadow)データ

    OptixProgramGroup radiance_hit_group = 0;           //!< hitgroup(radiance)データ
    OptixProgramGroup shadow_hit_group = 0;             //!< hitgroup(shadow)データ

    CUstream stream = 0;                                //!< ストリープ
    LaunchParam host_params;                            //!< ホスト側の開始パラメータ   
    LaunchParam* device_params;                         //!< デバイス側の開始パラメータ

    OptixShaderBindingTable sbt = {};                   //!< sbtデータ
    InitParam init_param = {};                          //!< 初期パラメータ

    CudaBuffer sky_solar_data = {};

    CUmodule cuda_module = 0;
    CUfunction cuda_tonmap = 0;
    CUfunction cuda_copybuffer = 0;
    CUfunction cuda_atrous_wavelet = 0;
    CUfunction cuda_particle = 0;
    CUfunction cuda_lensflare = 0;

    CudaBuffer albedo;
    CudaBuffer normal;
    CudaBuffer position;
    CudaBuffer sdr;
    CudaBuffer hdr;
    CudaBuffer accume_hdr;
    CudaBuffer hdr_tmp;

    OutputType output_type = OutputType::Albedo;
    int32_t wavelet_sample = 5;
    float color_sigma = 0.125f;
    float normal_sigma = 0.125f;
    float position_sigma = 0.125f;
    float albedo_sigma = 0.125f;
    float color_sigma_scale = 1.0f;
    bool enable_denoise = true;


    OptixDenoiser m_denoiser = nullptr;
    OptixDenoiserParams m_params = {};
    OptixDenoiserGuideLayer m_guide_layer = {};
    OptixDenoiserLayer m_layer = {};
    uint32_t m_scratch_size = 0;
    uint32_t m_state_size = 0;
    CudaBuffer m_state;
    CudaBuffer m_scratch;

    CudaBuffer m_materials;
    CudaBuffer m_lights;

    CUdeviceptr m_update_buffer = 0;
    CUdeviceptr m_update_compact_buffer = 0;

    ParticleSystem m_particle_system = {};
    std::vector<HitGroupRecord> hitgroup_records;

    LensSystem lens_system = {};
    CudaBuffer lens_system_buffer = {};
    CudaBuffer lens_system_param_buffer = {};
    CudaBuffer lens_system_rgb_buffer = {};
    CudaBuffer lens_sytem_blade_positions = {};
    CudaLensParam m_lens_param;

    std::unordered_map<std::string, Animation> animationTable;
    int32_t light_instance_id = 0;
};

// コンストラクタ
SampleScene::SampleScene()
    :m_impl(new Implement)
{

}

// デストラクタ
SampleScene::~SampleScene() 
{

}

// シーンをロード
void SampleScene::LoadScene(GraphicsContext& context)
{
    // sibenikをロードして作成
    
    std::string filepath = GetDirectoryWithPackage() + "\\model\\sibenik\\";
    std::string modelfile = filepath + "sibenik.model";
    std::ifstream ifs(modelfile.c_str(), std::ios::binary);
    if (!ifs.is_open())
    {
        ASSERT(false);
        return;
    }
    data::Scene sibenik_object = {};
    cereal::BinaryInputArchive input(ifs);
    input(sibenik_object);
    std::stringstream ss;
    
    for (size_t i = 0; i < sibenik_object.textures.size(); i++)
    {
        data::Texture& textrue = sibenik_object.textures.at(i);
        std::filesystem::path tmp_path = textrue.path;
        tmp_path = tmp_path.filename();
        textrue.path = filepath + tmp_path.string();
    }
    CreateObject(sibenik_object, context, m_impl->resources);
    
    for (size_t i = 0; i < m_impl->resources.material_table.size(); i++)
    {
        auto& material = m_impl->resources.material_table.at(i);
        if (i == 4) 
        {
            material.roughness = 0.0f;
        }
        else 
        {
            material.roughness = 0.3f;
        }
    }

    {
        CudaLight light;
        light.position = make_float3(-18.2688, -13.6746, -6.0186);
        light.radius = 0.1f;
        light.type = LightType::Sphere;
        light.emission = make_float3(1.0f, 1.0f, 1.0f);
        m_impl->light_instance_id = GenerateSphereLight(context, "sphere_light0", light, m_impl->resources);
    }
    


    //GenerateBSDFSample(context, m_impl->resources);
    //GenerateCornelBoxSample(context, m_impl->resources);

    {
        Animation light_animation;
        light_animation.AddData(make_float3(-9.48363, -13.6729, -5.93847));
        light_animation.AddData(make_float3(-2.54434, -13.0859, -5.45067));
        light_animation.AddData(make_float3(3.38645, -8.98339, -5.80825));
        light_animation.AddData(make_float3(11.2452, -4.117, -1.5f));
        m_impl->animationTable["light"] = light_animation;
    }

    {
        Animation camera_animation;
        camera_animation.AddData(make_float3(-2.65259, -12.8317, 0.177099));
        camera_animation.AddData(make_float3(-2.65259, -12.8317, 0.177099));
        camera_animation.AddData(make_float3(-2.65259, -12.8317, 0.177099));
        camera_animation.AddData(make_float3(-2.65259, -12.8317, 0.177099));
        m_impl->animationTable["camera_pos"] = camera_animation;
    }

    {
        Animation camera_animation;
        camera_animation.AddData(make_float3(-7.40543, -12.5602, -4.03187));
        camera_animation.AddData(make_float3(-0.796426, -12.7746, -5.89992));
        camera_animation.AddData(make_float3(1.00842, -10.0808, -4.22835));
        camera_animation.AddData(make_float3(11.2452, -5.117, -1.5f));
        m_impl->animationTable["camera_target"] = camera_animation;
    }
    return;
}

void SampleScene::SetupObject(GraphicsContext& context)
{
    // オブジェクトをセットアップ
    size_t material_count = m_impl->resources.material_table.size();
    size_t material_byte_size = material_count * sizeof(CudaMaterial);
    m_impl->m_materials.CreateBuffer(context, material_byte_size, data::ValueType::Invalid, data::ElementType::Invalid, "material_ids", m_impl->resources.material_table.data());

    size_t light_count = m_impl->resources.light_table.size();
    size_t light_byte_size = light_count * sizeof(CudaLight);
    m_impl->m_lights.CreateBuffer(context, light_byte_size, data::ValueType::Invalid, data::ElementType::Invalid, "light_table", m_impl->resources.light_table.data());

    CreatRootInstance(context, m_impl->resources, m_impl->m_update_buffer, m_impl->m_update_compact_buffer);
}

// プログラムデータを作成
void SampleScene::CreateModule(GraphicsContext& context) 
{
    // パイプラインオプションを設定
    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    m_impl->pipeline_option.usesMotionBlur = false;
    m_impl->pipeline_option.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

    // ペイロードデータは32bit x 2をのポインタでやり取りするので2つ
    m_impl->pipeline_option.numPayloadValues = 2;

    // レイは2種類使用するので2
    m_impl->pipeline_option.numAttributeValues = 2;

    // 開始パラメータの変数名はparams
    m_impl->pipeline_option.pipelineLaunchParamsVariableName = "params";

    // シェーダのバイナリデータを読み込みプログラムデータを作成
    DefaultFileSystem filesystem = {};
    std::string program_path = GetDirectoryWithPackage() + "\\" + "program\\" + "rgb_path_tracer.optixir";
    IBlobPtr blob = filesystem.ReadFile(program_path);
    if (!blob || !blob->data()) 
    {
        ASSERT(false);
        return;
    }

    context.CreateProgramModule(module_options, m_impl->pipeline_option, blob, &m_impl->program_module);
}

// プログラムグループを作成
void SampleScene::CreateProgramGroups(GraphicsContext& context) 
{
    OptixProgramGroupOptions  program_group_options = {};
    context.CreateProgramGroup(&m_impl->program_module, "__raygen__rg", OPTIX_PROGRAM_GROUP_KIND_RAYGEN, program_group_options, &m_impl->raygen_prog_group);
    context.CreateProgramGroup(&m_impl->program_module, "__miss__radiance", OPTIX_PROGRAM_GROUP_KIND_MISS, program_group_options, &m_impl->radiance_miss_group);
    context.CreateProgramGroup(&m_impl->program_module, "__miss__shadow", OPTIX_PROGRAM_GROUP_KIND_MISS, program_group_options, &m_impl->shadow_miss_group);
    context.CreateProgramGroup(&m_impl->program_module, "__closesthit__radiance", OPTIX_PROGRAM_GROUP_KIND_HITGROUP, program_group_options, &m_impl->radiance_hit_group);
    context.CreateProgramGroup(&m_impl->program_module, "__closesthit__shadow", OPTIX_PROGRAM_GROUP_KIND_HITGROUP, program_group_options, &m_impl->shadow_hit_group);
    //context.CreateProgramHitGroup(&m_impl->program_module, "__closesthit__radiance", "__anyhit__radiance", program_group_options, &m_impl->radiance_hit_group);
    //context.CreateProgramHitGroup(&m_impl->program_module, "__closesthit__shadow", "__anyhit__shadow", program_group_options, &m_impl->shadow_hit_group);
}

// パイプラインを作成
void SampleScene::CreatePipeline(GraphicsContext& context) 
{
    std::vector<OptixProgramGroup> program_groups =
    {
        m_impl->raygen_prog_group,
        m_impl->radiance_miss_group,
        m_impl->shadow_miss_group,
        m_impl->radiance_hit_group,
        m_impl->shadow_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;
    context.CreatePipeline(program_groups, pipeline_link_options, m_impl->pipeline_option, &m_impl->pipeline);
}

// SBG(ShaderBindTable)を作成
void SampleScene::CreateSBT(GraphicsContext& context) 
{
    // raygenプログラムに紐づけるデータを作成
    CUdeviceptr  d_raygen_record = 0;
    {
        RayGenRecord rg_sbt = {};
        context.PackRecordHeader((void*)m_impl->raygen_prog_group, &rg_sbt);
        std::vector<RayGenRecord> rg_sbts = { rg_sbt };
        context.AllocateAndMemcpySBT(d_raygen_record, sizeof(RayGenRecord) * rg_sbts.size(), rg_sbts.data());
    }

    // missプログラムに紐づけるデータを作成
    CUdeviceptr  d_miss_records = 0;
    const size_t miss_record_size = sizeof(MissRecord);
    {
        MissRecord ms_sbt[RAY_TYPE_COUNT];

        // Radiance Ray用
        context.PackRecordHeader((void*)m_impl->radiance_miss_group, &ms_sbt[0]);
        ms_sbt[RAY_TYPE_RADIANCE].data.bg_color = make_float4(0.0f);
        ms_sbt[RAY_TYPE_RADIANCE].data.env_tex = m_impl->resources.envrionment_texture.texObj;

        // Occlusion Ray用
        context.PackRecordHeader((void*)m_impl->shadow_miss_group, &ms_sbt[1]);
        ms_sbt[RAY_TYPE_OCCLUSION].data.bg_color = make_float4(0.0f);
        ms_sbt[RAY_TYPE_OCCLUSION].data.env_tex = m_impl->resources.envrionment_texture.texObj;

        context.AllocateAndMemcpySBT(d_miss_records, sizeof(MissRecord) * RAY_TYPE_COUNT, ms_sbt);
    }

    // hit groupに紐づけるデータを作成
    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    const size_t hitgroup_count = RAY_TYPE_COUNT * m_impl->resources.geometry_as_table.size();
    {
        // hit groupはインスタンス数 * レイデータ分作る
        m_impl->hitgroup_records.resize(hitgroup_count);

        for (size_t i = 0; i < m_impl->resources.instance_as_table.size(); i++)
        {
            float sbt_value = static_cast<float>(i) / static_cast<float>(m_impl->hitgroup_records.size());
            const InstanceAS& instance_as = m_impl->resources.instance_as_table.at(i);
            const GeometryAS& geometry_as = m_impl->resources.geometry_as_table.at(instance_as.bind_geometry_as_id);
            const CudaBuffer& vertex_buffer = m_impl->resources.buffer_table.at(geometry_as.vertex_buffer_id);
            const CudaBuffer& normal_buffer = m_impl->resources.buffer_table.at(geometry_as.normal_buffer_id);
            const CudaBuffer& texcoord_buffer = m_impl->resources.buffer_table.at(geometry_as.texcoord_buffer_id);
            const CudaBuffer& material_buffer = m_impl->resources.buffer_table.at(geometry_as.material_buffer_id);
            const CudaBuffer& index_buffer = m_impl->resources.buffer_table.at(geometry_as.index_buffer_id);
            const glm::mat4x4& world = instance_as.transform.world;
            const glm::mat4x4& normal = instance_as.transform.normal;

            //const CudaTexture& albedo = m_impl->resources.texture_table.at(0);
            // Radiance Ray
            {
                const int32_t sbt_index = static_cast<int32_t>(i * RAY_TYPE_COUNT + RAY_TYPE_RADIANCE);
                HitGroupRecord& hitgroup_record = m_impl->hitgroup_records.at(sbt_index);
                context.PackRecordHeader((void*)m_impl->radiance_hit_group, &hitgroup_record);

                hitgroup_record.data.vertices = reinterpret_cast<float3*>(vertex_buffer.buffer);
                hitgroup_record.data.normals = reinterpret_cast<float3*>(normal_buffer.buffer);
                hitgroup_record.data.texcoords = reinterpret_cast<float2*>(texcoord_buffer.buffer);
                hitgroup_record.data.material_ids = reinterpret_cast<unsigned int*>(material_buffer.buffer);
                hitgroup_record.data.indices = reinterpret_cast<unsigned int*>(index_buffer.buffer);
                hitgroup_record.data.sbt_index = sbt_value;
                hitgroup_record.data.local0 = make_float4(world[0][0], world[0][1], world[0][2], world[0][3]);
                hitgroup_record.data.local1 = make_float4(world[1][0], world[1][1], world[1][2], world[1][3]);
                hitgroup_record.data.local2 = make_float4(world[2][0], world[2][1], world[2][2], world[2][3]);
                hitgroup_record.data.normal0 = make_float4(normal[0][0], normal[0][1], normal[0][2], normal[0][3]);
                hitgroup_record.data.normal1 = make_float4(normal[1][0], normal[1][1], normal[1][2], normal[1][3]);
                hitgroup_record.data.normal2 = make_float4(normal[2][0], normal[2][1], normal[2][2], normal[2][3]);
            }

            // Occlusion Ray
            {
                const int32_t sbt_index = static_cast<int32_t>(i * RAY_TYPE_COUNT + RAY_TYPE_OCCLUSION);
                HitGroupRecord& hitgroup_record = m_impl->hitgroup_records.at(sbt_index);
                context.PackRecordHeader((void*)m_impl->shadow_hit_group, &hitgroup_record);

                hitgroup_record.data.vertices = reinterpret_cast<float3*>(vertex_buffer.buffer);
                hitgroup_record.data.normals = reinterpret_cast<float3*>(normal_buffer.buffer);
                hitgroup_record.data.texcoords = reinterpret_cast<float2*>(texcoord_buffer.buffer);
                hitgroup_record.data.material_ids = reinterpret_cast<unsigned int*>(material_buffer.buffer);
                hitgroup_record.data.indices = reinterpret_cast<unsigned int*>(index_buffer.buffer);
                hitgroup_record.data.sbt_index = sbt_value;
                hitgroup_record.data.local0 = make_float4(world[0][0], world[0][1], world[0][2], world[0][3]);
                hitgroup_record.data.local1 = make_float4(world[1][0], world[1][1], world[1][2], world[1][3]);
                hitgroup_record.data.local2 = make_float4(world[2][0], world[2][1], world[2][2], world[2][3]);
                hitgroup_record.data.normal0 = make_float4(normal[0][0], normal[0][1], normal[0][2], normal[0][3]);
                hitgroup_record.data.normal1 = make_float4(normal[1][0], normal[1][1], normal[1][2], normal[1][3]);
                hitgroup_record.data.normal2 = make_float4(normal[2][0], normal[2][1], normal[2][2], normal[2][3]);
            }
        }
        context.AllocateAndMemcpySBT(d_hitgroup_records, sizeof(HitGroupRecord) * m_impl->hitgroup_records.size(), m_impl->hitgroup_records.data());
    }

    // 各種データを設定
    m_impl->sbt.raygenRecord = d_raygen_record;

    m_impl->sbt.missRecordBase = d_miss_records;
    m_impl->sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    m_impl->sbt.missRecordCount = RAY_TYPE_COUNT;

    m_impl->sbt.hitgroupRecordBase = d_hitgroup_records;
    m_impl->sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    m_impl->sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_count);
}

// 開始パラメータの設定
void SampleScene::InitLaunchParams(GraphicsContext& context, const InitParam& param) 
{
    m_impl->init_param = param;

    // 各種バッファのメモリ確保
    size_t float_byte_size = param.width * param.height * sizeof(float4);
    size_t uchar_byte_size = param.width * param.height * sizeof(uchar4);
    m_impl->albedo.CreateBuffer(context, float_byte_size, data::ValueType::Float, data::ElementType::Vector4, "albedo", nullptr);
    m_impl->normal.CreateBuffer(context, float_byte_size, data::ValueType::Float, data::ElementType::Vector4, "normal", nullptr);
    m_impl->position.CreateBuffer(context, float_byte_size, data::ValueType::Float, data::ElementType::Vector4, "position", nullptr);
    m_impl->hdr.CreateBuffer(context, float_byte_size, data::ValueType::Float, data::ElementType::Vector4, "hdr", nullptr);
    m_impl->accume_hdr.CreateBuffer(context, float_byte_size, data::ValueType::Float, data::ElementType::Vector4, "accume_hdr", nullptr);
    m_impl->hdr_tmp.CreateBuffer(context, float_byte_size, data::ValueType::Float, data::ElementType::Vector4, "hdr_tmp", nullptr);
    m_impl->sdr.CreateBuffer(context, uchar_byte_size, data::ValueType::Uint8, data::ElementType::Vector4, "sdr", nullptr);

    m_impl->host_params.albedo_buffer = reinterpret_cast<float4*>(m_impl->albedo.buffer);
    m_impl->host_params.normal_buffer = reinterpret_cast<float4*>(m_impl->normal.buffer);
    m_impl->host_params.position_buffer = reinterpret_cast<float4*>(m_impl->position.buffer);
    m_impl->host_params.accum_buffer = reinterpret_cast<float4*>(m_impl->accume_hdr.buffer);
    m_impl->host_params.frame_buffer = reinterpret_cast<float4*>(m_impl->hdr.buffer);

    m_impl->host_params.subframe_index = 0u;
    m_impl->host_params.samples_per_launch = param.sample_per_launch;
    m_impl->host_params.max_depth = param.max_depth;
    m_impl->host_params.min_dist = 0.00001f;
    m_impl->host_params.max_dist = 1e16f;

    m_impl->host_params.direct_light.dir = Normalize(make_float3(param.sun_dir[0], param.sun_dir[1], param.sun_dir[2]));
    m_impl->host_params.direct_light.emission = make_float3(1.0f, 1.0f, 1.0f);
    m_impl->host_params.direct_light.sky_intensity = param.sky_intensity;

    m_impl->host_params.handle = m_impl->resources.root_instance_as.handle;

    m_impl->host_params.debug.debug_mode = DebugMode_None;
    m_impl->host_params.debug.color = make_float3(1.0f, 1.0f, 1.0f);
    m_impl->host_params.materials = reinterpret_cast<Material*>(m_impl->m_materials.buffer);
    m_impl->host_params.lights = reinterpret_cast<AnyLight*>(m_impl->m_lights.buffer);
    m_impl->host_params.light_count = static_cast<uint32_t>(m_impl->resources.light_table.size());
    m_impl->host_params.debug.russian_roulette = param.russian_roulette;

    // 更新用にストリームデータを作成
    CUDA_CHECK(cudaStreamCreate(&m_impl->stream));

    // 開始パラメータのメモリ確保
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_impl->device_params), sizeof(LaunchParam)));
}

void SampleScene::InitDenoiser(GraphicsContext& context, const InitParam& param)
{
    OptixDenoiserOptions options = {};
    options.guideAlbedo = 1;
    options.guideNormal = 1;

    OptixDenoiserModelKind model_kind = OPTIX_DENOISER_MODEL_KIND_HDR;
    context.CreateOptixDenoiser(options, model_kind, m_impl->m_denoiser);

    OptixDenoiserSizes denoiser_sizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_impl->m_denoiser, param.width, param.height, &denoiser_sizes));
    m_impl->m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withoutOverlapScratchSizeInBytes);
    m_impl->m_state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);

    m_impl->m_scratch.CreateBuffer(context, m_impl->m_scratch_size, data::ValueType::Uint8, data::ElementType::Vector1, "denoiser_scratch", nullptr);
    m_impl->m_state.CreateBuffer(context, m_impl->m_state_size, data::ValueType::Uint8, data::ElementType::Vector1, "denoiser_state", nullptr);

    OPTIX_CHECK(optixDenoiserSetup(
        m_impl->m_denoiser, nullptr, param.width, param.height, 
        m_impl->m_state.buffer, m_impl->m_state_size, m_impl->m_scratch.buffer, m_impl->m_scratch_size));

    m_impl->m_params.hdrIntensity = reinterpret_cast<CUdeviceptr>(nullptr);
    m_impl->m_params.hdrAverageColor = reinterpret_cast<CUdeviceptr>(nullptr);


    m_impl->m_guide_layer.albedo.width = param.width;
    m_impl->m_guide_layer.albedo.height = param.height;
    m_impl->m_guide_layer.albedo.rowStrideInBytes = param.width * sizeof(float4);
    m_impl->m_guide_layer.albedo.pixelStrideInBytes = sizeof(float4);
    m_impl->m_guide_layer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    m_impl->m_guide_layer.albedo.data = m_impl->albedo.buffer;


    m_impl->m_guide_layer.normal.width = param.width;
    m_impl->m_guide_layer.normal.height = param.height;
    m_impl->m_guide_layer.normal.rowStrideInBytes = param.width * sizeof(float4);
    m_impl->m_guide_layer.normal.pixelStrideInBytes = sizeof(float4);
    m_impl->m_guide_layer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    m_impl->m_guide_layer.normal.data = m_impl->normal.buffer;

    m_impl->m_layer.input.width = param.width;
    m_impl->m_layer.input.height = param.height;
    m_impl->m_layer.input.rowStrideInBytes = param.width * sizeof(float4);
    m_impl->m_layer.input.pixelStrideInBytes = sizeof(float4);
    m_impl->m_layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    m_impl->m_layer.input.data = m_impl->hdr_tmp.buffer;

    m_impl->m_layer.output.width = param.width;
    m_impl->m_layer.output.height = param.height;
    m_impl->m_layer.output.rowStrideInBytes = param.width * sizeof(float4);
    m_impl->m_layer.output.pixelStrideInBytes = sizeof(float4);
    m_impl->m_layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    m_impl->m_layer.output.data = m_impl->hdr.buffer;

}

void SampleScene::CreateCudaModule(GraphicsContext& context)
{
    DefaultFileSystem filesystem = {};
    std::string program_path = GetDirectoryWithPackage() + "\\" + "program\\" + "post_process.ptx";
    IBlobPtr blob = filesystem.ReadFile(program_path);
    if (!blob || !blob->data())
    {

        ASSERT(false);
        return;
    }

    context.CreateCudaModule(blob, m_impl->cuda_module);
    context.GetFunctionKernel(m_impl->cuda_module, "launch_tonemap", m_impl->cuda_tonmap);
    context.GetFunctionKernel(m_impl->cuda_module, "launch_copybuffer", m_impl->cuda_copybuffer);
    context.GetFunctionKernel(m_impl->cuda_module, "launch_atrous_wavelet", m_impl->cuda_atrous_wavelet);
    context.GetFunctionKernel(m_impl->cuda_module, "launch_particle", m_impl->cuda_particle);
    context.GetFunctionKernel(m_impl->cuda_module, "launch_lensflare", m_impl->cuda_lensflare);
}

void SampleScene::UpdateAnimation(int32_t framecount, Camera& camera)
{

    float3 light_position;
    if (m_impl->animationTable["light"].GetPosition(framecount, light_position))
    {
        auto& instanecAS = m_impl->resources.instance_as_table.at(m_impl->light_instance_id);
        instanecAS.transform.translate.x = light_position.x;
        instanecAS.transform.translate.y = light_position.y;
        instanecAS.transform.translate.z = light_position.z;
        instanecAS.transform.update();
        UpdateTransform(instanecAS);

        m_impl->resources.light_table.at(0).position.x = light_position.x;
        m_impl->resources.light_table.at(0).position.y = light_position.y;
        m_impl->resources.light_table.at(0).position.z = light_position.z;
    }
    float3 camera_target;
    if (m_impl->animationTable["camera_target"].GetPosition(framecount, camera_target))
    {
        camera.SetRegardPosition(camera_target);
        camera.drity = true;
    }

    float3 camera_position;
    if (m_impl->animationTable["camera_pos"].GetPosition(framecount, camera_position))
    {
        camera.SetPosition(camera_position);
        camera.drity = true;
    }
}

void SampleScene::UpdateParticle(GraphicsContext& context, uint32_t framecount)
{
    //m_impl->m_particle_system.UpdateParticleLight(context, m_impl->resources, m_impl->cuda_particle, framecount);
}


void SampleScene::UpdateLaunchParam(SdrPixelBuffer& output_buffer, InitParam& param)
{
    m_impl->host_params.width = output_buffer.GetWidth();
    m_impl->host_params.height = output_buffer.GetHeight();
    m_impl->host_params.samples_per_launch = param.sample_per_launch;
    m_impl->host_params.max_depth = param.max_depth;
    m_impl->host_params.debug.debug_mode = param.debug_mode;
    m_impl->host_params.debug.color = make_float3(param.debug_color[0], param.debug_color[1], param.debug_color[2]);
    m_impl->host_params.debug.russian_roulette = param.russian_roulette;
    m_impl->output_type = param.output_type;
    m_impl->enable_denoise = param.enable_denoise;
    m_impl->init_param = param;
    m_impl->init_param.lens_param.width = param.width;
    m_impl->init_param.lens_param.height = param.height;
    param.lens_param.dirty = false;
}

bool SampleScene::UpdateObject(GraphicsContext& context)
{
    for (size_t i = 0; i < m_impl->resources.instance_as_table.size(); i++) 
    {
        auto& instanceAS = m_impl->resources.instance_as_table.at(i);
        UpdateTransform(instanceAS);
    }
    return true;
}

bool SampleScene::UpdateLight(GraphicsContext& context)
{
    context.UpdateBuffer(m_impl->m_lights.buffer, m_impl->m_lights.byte_size, m_impl->resources.light_table.data());
    return true;
}

bool SampleScene::UpdateLensSystem(GraphicsContext& context)
{
    if (m_impl->init_param.lens_param.dirty)
    {
        m_impl->lens_system.SetupParam(m_impl->init_param.lens_param);
        auto& systems = m_impl->lens_system.GetSystem();
        auto& cuda_lens_param = m_impl->lens_system.GetCudaLensParam();
        auto& rgb_param = m_impl->lens_system.GetRGBParam();
        auto& blade_positions = m_impl->lens_system.GetBladePositions();
        size_t byte_size = 0;

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_impl->lens_system_buffer.buffer)));
        byte_size = sizeof(CudaPolySystem33) * systems.size();
        m_impl->lens_system_buffer.CreateBuffer(context, byte_size, data::ValueType::Invalid, data::ElementType::Invalid, "lens_system", systems.data());

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_impl->lens_system_param_buffer.buffer)));
        byte_size = sizeof(CudaLensParam);
        m_impl->lens_system_param_buffer.CreateBuffer(context, byte_size, data::ValueType::Invalid, data::ElementType::Invalid, "lens_system_param", &cuda_lens_param);

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_impl->lens_system_rgb_buffer.buffer)));
        byte_size = sizeof(float) * rgb_param.size();
        m_impl->lens_system_rgb_buffer.CreateBuffer(context, byte_size, data::ValueType::Invalid, data::ElementType::Invalid, "lens_system_rgb", rgb_param.data());
        
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_impl->lens_sytem_blade_positions.buffer)));
        byte_size = sizeof(float) * blade_positions.size();
        m_impl->lens_sytem_blade_positions.CreateBuffer(context, byte_size, data::ValueType::Invalid, data::ElementType::Invalid, "lens_system_rgb", blade_positions.data());
    }
    return false;
}

bool SampleScene::UpdateCamera(Camera& camera)
{
    if (camera.drity)
    {
        LaunchParam& params = m_impl->host_params;
        camera.SetAspect(static_cast<float>(params.width) / static_cast<float>(params.height));
        params.camera.eye = camera.GetPosition();
        camera.GetCameraSpace(params.camera.U, params.camera.V, params.camera.W);
        camera.drity = false;
        return true;
    }
    return false;
}

bool SampleScene::UpdateResize(SdrPixelBuffer& output_buffer, bool resize_dirty)
{
    if (resize_dirty)
    {
        LaunchParam& params = m_impl->host_params;
        output_buffer.Resize(params.width, params.height);

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.accum_buffer), params.width * params.height * sizeof(float4)));

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.albedo_buffer)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.albedo_buffer), params.width * params.height * sizeof(float4)));

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.normal_buffer)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.normal_buffer), params.width * params.height * sizeof(float4)));

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.position_buffer)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.position_buffer), params.width * params.height * sizeof(float4)));
        return true;
    }
    return false;
}

bool SampleScene::UpdateSky(InitParam& param)
{
    if (param.sky_dirty)
    {
        m_impl->host_params.direct_light.dir = Normalize(make_float3(param.sun_dir[0], param.sun_dir[1], param.sun_dir[2]));
        m_impl->host_params.direct_light.emission = make_float3(param.sun_emission[0], param.sun_emission[1], param.sun_emission[2]);
#if ENABLE_ARHOSEK_SKY
        m_impl->host_params.direct_light.sky_intensity = param.sky_intensity;
        float solar_elevation = 0.0;
        {
            float3 p = m_impl->host_params.direct_light.dir;
            float r = Length(p);
            float r_xz = Length(make_float2(p.x, p.z));
            float theta = acos(r_xz / r) * (p.y < 0.0f ? -1.0f : 1.0f);
            solar_elevation = theta;
        }
        arhosek_rgb_skymodelstate_alloc_init(m_impl->host_params.direct_light.sky_state, param.atmospheric_turbidity, param.ground_albedo, solar_elevation);
#endif
        const float3 K = make_float3(0.686282f, 0.677739f, 0.663365f);
        const float3 lambda = make_float3(680e-9f, 550e-9f, 440e-9f);

        m_impl->host_params.direct_light.sky_state.betaR0 = ComputeCoefficientRayleigh(lambda);
        m_impl->host_params.direct_light.sky_state.betaM0 = ComputeCoefficientMie(lambda, K, exp(100.0f));
        param.sky_dirty = false;
        return true;
    }
    return false;
}

void SampleScene::UpdateIAS(GraphicsContext& context, bool is_update)
{
    if (is_update) 
    {
        UpdateRootInstance(context, m_impl->resources, m_impl->stream, m_impl->m_update_buffer, m_impl->m_update_compact_buffer);

        /*
        for (size_t i = 0; i < m_impl->resources.instance_as_table.size(); i++)
        {
            const InstanceAS& instance_as = m_impl->resources.instance_as_table.at(i);
            const glm::mat4x4& world = instance_as.transform.world;
            const glm::mat4x4& normal = instance_as.transform.normal;

            // Radiance Ray
            {
                const int32_t sbt_index = static_cast<int32_t>(i * RAY_TYPE_COUNT + RAY_TYPE_RADIANCE);
                HitGroupRecord& hitgroup_record = m_impl->hitgroup_records.at(sbt_index);
                context.PackRecordHeader((void*)m_impl->radiance_hit_group, &hitgroup_record);

                hitgroup_record.data.local0 = make_float4(world[0][0], world[0][1], world[0][2], world[0][3]);
                hitgroup_record.data.local1 = make_float4(world[1][0], world[1][1], world[1][2], world[1][3]);
                hitgroup_record.data.local2 = make_float4(world[2][0], world[2][1], world[2][2], world[2][3]);
                hitgroup_record.data.normal0 = make_float4(normal[0][0], normal[0][1], normal[0][2], normal[0][3]);
                hitgroup_record.data.normal1 = make_float4(normal[1][0], normal[1][1], normal[1][2], normal[1][3]);
                hitgroup_record.data.normal2 = make_float4(normal[2][0], normal[2][1], normal[2][2], normal[2][3]);
            }

            // Occlusion Ray
            {
                const int32_t sbt_index = static_cast<int32_t>(i * RAY_TYPE_COUNT + RAY_TYPE_OCCLUSION);
                HitGroupRecord& hitgroup_record = m_impl->hitgroup_records.at(sbt_index);
                context.PackRecordHeader((void*)m_impl->shadow_hit_group, &hitgroup_record);
                hitgroup_record.data.local0 = make_float4(world[0][0], world[0][1], world[0][2], world[0][3]);
                hitgroup_record.data.local1 = make_float4(world[1][0], world[1][1], world[1][2], world[1][3]);
                hitgroup_record.data.local2 = make_float4(world[2][0], world[2][1], world[2][2], world[2][3]);
                hitgroup_record.data.normal0 = make_float4(normal[0][0], normal[0][1], normal[0][2], normal[0][3]);
                hitgroup_record.data.normal1 = make_float4(normal[1][0], normal[1][1], normal[1][2], normal[1][3]);
                hitgroup_record.data.normal2 = make_float4(normal[2][0], normal[2][1], normal[2][2], normal[2][3]);
            }
        }
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_impl->sbt.hitgroupRecordBase), m_impl->hitgroup_records.data(), sizeof(HitGroupRecord) * m_impl->hitgroup_records.size(), cudaMemcpyHostToDevice));
        */
    }
}
void  SampleScene::UpdateSubframe(bool is_reset)
{
    m_impl->host_params.subframe_index++;
    if (m_impl->host_params.subframe_index > std::numeric_limits<uint32_t>::max() - 1 || is_reset)
    {
        m_impl->host_params.subframe_index = 0;
    }
}

void SampleScene::LaunchOptixKernel(LaunchArg& arg) 
{
    if (arg.output_buffer == nullptr)
    {
        return;
    }

    // ホスト側のデータをデバイス側にコピー
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(m_impl->device_params),
        &m_impl->host_params,
        sizeof(LaunchParam),
        cudaMemcpyHostToDevice,
        m_impl->stream));

    // 開始
    OPTIX_CHECK(optixLaunch(
        m_impl->pipeline,
        m_impl->stream,
        reinterpret_cast<CUdeviceptr>(m_impl->device_params),
        sizeof(LaunchParam),
        &m_impl->sbt,
        m_impl->host_params.width,
        m_impl->host_params.height,
        1));

    CUDA_SYNC_CHECK();
}

void SampleScene::TonemapPass(int width, int height)
{
    int numElements = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    dim3 cudaBlockSize(threadsPerBlock, 1, 1);
    dim3 cudaGridSize(blocksPerGrid, 1, 1);

    void* arr[] =
    {
        reinterpret_cast<void*>(&m_impl->hdr.buffer),
        reinterpret_cast<void*>(&m_impl->sdr.buffer),
        reinterpret_cast<void*>(&numElements),
    };

    CUDA_CHECK_ERROR(cuLaunchKernel(m_impl->cuda_tonmap, cudaGridSize.x, cudaGridSize.y, cudaGridSize.z,
        cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, 0, nullptr, arr, 0));

    CUDA_CHECK_ERROR(cuCtxSynchronize());
}

void SampleScene::AtrousWaveletPass(int width, int height)
{
    if (!m_impl->enable_denoise)
    {
        return;
    }
    int numElements = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    dim3 cudaBlockSize(threadsPerBlock, 1, 1);
    dim3 cudaGridSize(blocksPerGrid, 1, 1);

    float color_sigma = m_impl->color_sigma;
    float normal_sigma = m_impl->normal_sigma;
    float position_sigma = m_impl->position_sigma;
    float albedo_sigma = m_impl->albedo_sigma;
    float color_sigma_scale = m_impl->color_sigma_scale;
    int32_t step_scale = 1;
    void* arr[13] = {};
    arr[0] = reinterpret_cast<void*>(&m_impl->hdr.buffer);
    arr[1] = reinterpret_cast<void*>(&m_impl->albedo.buffer);
    arr[2] = reinterpret_cast<void*>(&m_impl->normal.buffer);
    arr[3] = reinterpret_cast<void*>(&m_impl->position.buffer);
    arr[4] = reinterpret_cast<void*>(&m_impl->hdr_tmp.buffer);
    arr[5] = reinterpret_cast<void*>(&numElements);
    arr[6] = reinterpret_cast<void*>(&width);
    arr[7] = reinterpret_cast<void*>(&height);
    arr[8] = reinterpret_cast<void*>(&color_sigma);
    arr[9] = reinterpret_cast<void*>(&normal_sigma);
    arr[10] = reinterpret_cast<void*>(&position_sigma);
    arr[11] = reinterpret_cast<void*>(&albedo_sigma);
    arr[12] = reinterpret_cast<void*>(&step_scale);
    for (int32_t i = 0 ; i < m_impl->wavelet_sample; i++)
    {
        if (i % 2 != 0) 
        {
            arr[0] = reinterpret_cast<void*>(&m_impl->hdr_tmp.buffer);
            arr[4] = reinterpret_cast<void*>(&m_impl->hdr.buffer);
        }
        else 
        {
            arr[0] = reinterpret_cast<void*>(&m_impl->hdr.buffer);
            arr[4] = reinterpret_cast<void*>(&m_impl->hdr_tmp.buffer);
        }

        step_scale = 1 << i;
        color_sigma_scale = std::powf(2.0f, (float)i);

        CUDA_CHECK_ERROR(cuLaunchKernel(m_impl->cuda_atrous_wavelet, cudaGridSize.x, cudaGridSize.y, cudaGridSize.z,
            cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, 0, nullptr, arr, 0));

        CUDA_CHECK_ERROR(cuCtxSynchronize());
    }
}

void SampleScene::OptixDenoiserPass()
{
    if (!m_impl->enable_denoise) 
    {
        return;
    }
    cudaMemcpy(reinterpret_cast<void*>(m_impl->hdr_tmp.buffer), reinterpret_cast<void*>(m_impl->hdr.buffer), m_impl->hdr.byte_size, cudaMemcpyDeviceToDevice);

    OPTIX_CHECK(optixDenoiserInvoke(
        m_impl->m_denoiser, 
        nullptr, 
        &m_impl->m_params,
        m_impl->m_state.buffer, 
        m_impl->m_state_size,
        &m_impl->m_guide_layer, 
        &m_impl->m_layer, 
        1, 0, 0,
        m_impl->m_scratch.buffer,
        m_impl->m_scratch_size));
}

void SampleScene::LensSystemPass(int width, int height)
{
    if (!m_impl->init_param.lens_param.enable_lensflare)
    {
        return;
    }

#if 1
    cudaMemcpy(reinterpret_cast<void*>(m_impl->hdr_tmp.buffer), reinterpret_cast<void*>(m_impl->hdr.buffer), m_impl->hdr.byte_size, cudaMemcpyDeviceToDevice);
    int numElements = width * height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    void* arr[10];
    arr[0] = reinterpret_cast<void*>(&m_impl->hdr.buffer);
    arr[1] = reinterpret_cast<void*>(&m_impl->hdr_tmp.buffer);
    arr[2] = reinterpret_cast<void*>(&numElements);
    arr[3] = reinterpret_cast<void*>(&width);
    arr[4] = reinterpret_cast<void*>(&height);
    arr[5] = reinterpret_cast<void*>(&m_impl->lens_system_buffer.buffer);
    arr[6] = reinterpret_cast<void*>(&m_impl->init_param.lens_param.num_lambdas);
    arr[7] = reinterpret_cast<void*>(&m_impl->lens_system_rgb_buffer.buffer);
    arr[8] = reinterpret_cast<void*>(&m_impl->lens_system_param_buffer.buffer);
    arr[9] = reinterpret_cast<void*>(&m_impl->lens_sytem_blade_positions.buffer);

    dim3 cudaBlockSize(threadsPerBlock, 1, 1);
    dim3 cudaGridSize(blocksPerGrid, 1, 1);
    CUDA_CHECK_ERROR(cuLaunchKernel(m_impl->cuda_lensflare,
        cudaGridSize.x, cudaGridSize.y, cudaGridSize.z,
        cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, 
        0, nullptr, arr, 0));

    CUDA_CHECK_ERROR(cuCtxSynchronize());
#else 
    std::vector<float4> src;
    std::vector<float4> dst;

    src.resize(width * height);
    dst.resize(width * height);

    cudaMemcpy(src.data(), reinterpret_cast<void*>(m_impl->hdr.buffer), m_impl->hdr.byte_size, cudaMemcpyDeviceToHost);
    m_impl->lens_system.CalcLensSystemImage(src, dst);
    cudaMemcpy(reinterpret_cast<void*>(m_impl->hdr.buffer),dst.data(), m_impl->hdr.byte_size, cudaMemcpyHostToDevice);
#endif
}

void SampleScene::CopyBufferPass(int width, int height)
{
    if (m_impl->output_type == OutputType::Sdr)
    {
        return;
    }

    int numElements = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    dim3 cudaBlockSize(threadsPerBlock, 1, 1);
    dim3 cudaGridSize(blocksPerGrid, 1, 1);

    void* arr[3] = {nullptr, nullptr, nullptr};
    if (m_impl->output_type == OutputType::Albedo)
    {
        arr[0] = reinterpret_cast<void*>(&m_impl->albedo.buffer);
    }
    else if (m_impl->output_type == OutputType::Normal)
    {
        arr[0] = reinterpret_cast<void*>(&m_impl->normal.buffer);
    }
    else if (m_impl->output_type == OutputType::Position)
    {
        arr[0] = reinterpret_cast<void*>(&m_impl->position.buffer);
    }
    else if (m_impl->output_type == OutputType::Hdr)
    {
        arr[0] = reinterpret_cast<void*>(&m_impl->hdr.buffer);
    }

    arr[1] = reinterpret_cast<void*>(&m_impl->sdr.buffer);
    arr[2] = reinterpret_cast<void*>(&numElements);


    CUDA_CHECK_ERROR(cuLaunchKernel(m_impl->cuda_copybuffer, cudaGridSize.x, cudaGridSize.y, cudaGridSize.z,
        cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, 0, nullptr, arr, 0));

    CUDA_CHECK_ERROR(cuCtxSynchronize());
}

void SampleScene::LaunchCudaKernel(LaunchArg& arg)
{
    int width = arg.hdr_buffer->GetWidth();
    int height = arg.hdr_buffer->GetHeight();
    //AtrousWaveletPass(width, height);
    LensSystemPass(width, height);
    OptixDenoiserPass();
    TonemapPass(width, height);
    CopyBufferPass(width, height);
}

// 初期化
void SampleScene::Initalize(GraphicsContext& context, const InitParam& param) 
{
    LoadScene(context);
    SetupObject(context);
    CreateModule(context);
    CreateProgramGroups(context);
    CreatePipeline(context);
    CreateSBT(context);
    InitLaunchParams(context, param);
    CreateCudaModule(context);
    InitDenoiser(context, param);
}

// 更新
void SampleScene::UpdateState(GraphicsContext& context, SdrPixelBuffer& output_buffer, Camera& camera, bool resize_dirty, InitParam& param, uint32_t framecount)
{
    //UpdateParticle(context, framecount);
    bool is_updated = false;
    UpdateAnimation(framecount, camera);
    UpdateLaunchParam(output_buffer, param);
    is_updated |= UpdateObject(context);
    is_updated |= UpdateCamera(camera);
    //is_updated |= UpdateResize(output_buffer, resize_dirty);
    is_updated |= UpdateSky(param);
    UpdateLight(context);
    UpdateIAS(context, is_updated);
    UpdateLensSystem(context);
    UpdateSubframe(false);
}

void SampleScene::CopyOutputBuffer(LaunchArg& arg)
{
    // 出力バッファを設定
    uchar4* sdr_buffer = arg.output_buffer->Map();
    float4* hdr_buffer = arg.hdr_buffer->Map();

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sdr_buffer), reinterpret_cast<void*>(m_impl->sdr.buffer), m_impl->sdr.byte_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hdr_buffer), reinterpret_cast<void*>(m_impl->hdr.buffer), m_impl->hdr.byte_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice));

    arg.output_buffer->Unmap();
    arg.hdr_buffer->Unmap();

}

// 開始
void SampleScene::LaunchSubframe(LaunchArg& arg) 
{
    LaunchOptixKernel(arg);
    LaunchCudaKernel(arg);
    CopyOutputBuffer(arg);
}

// 終了処理
void SampleScene::CleanupState() 
{
    OPTIX_CHECK(optixPipelineDestroy(m_impl->pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(m_impl->raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(m_impl->radiance_miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(m_impl->radiance_hit_group));
    OPTIX_CHECK(optixProgramGroupDestroy(m_impl->shadow_hit_group));
    OPTIX_CHECK(optixProgramGroupDestroy(m_impl->shadow_miss_group));
    OPTIX_CHECK(optixModuleDestroy(m_impl->program_module));


    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_impl->sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_impl->sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_impl->sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_impl->device_params)));
}
} // namespace slug