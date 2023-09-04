#include "graphic/GeometryObject.hpp"
#include "graphic/SampleGeometryData.hpp"
#include <random>
namespace slug
{

struct MeshData 
{
    std::vector<float> position;
    std::vector<float> normal;
    std::vector<float> uv;
    std::vector<uint32_t> material_ids;
    std::vector<uint32_t> index;
    uint32_t vertex_count = 0;
    uint32_t index_count = 0;
};

void ConvertMaterial(data::Material& material, const CudaMaterial& cuda_material) 
{

    data::CustomParam4f diffuse;
    diffuse.name = "base_color";
    diffuse.parameter[0] = cuda_material.base_color[0];
    diffuse.parameter[1] = cuda_material.base_color[1];
    diffuse.parameter[2] = cuda_material.base_color[2];
    material.c_param4f.push_back(diffuse);

    data::CustomParam4f emission;
    emission.name = "emission";
    emission.parameter[0] = cuda_material.emission[0];
    emission.parameter[1] = cuda_material.emission[1];
    emission.parameter[2] = cuda_material.emission[2];
    material.c_param4f.push_back(emission);

    data::CustomParamf ior;
    ior.name = "ior";
    ior.parameter = cuda_material.ior;
    material.c_paramf.push_back(ior);


    data::CustomParamf relative_ior;
    relative_ior.name = "relative_ior";
    relative_ior.parameter = cuda_material.relative_ior;
    material.c_paramf.push_back(relative_ior);

    data::CustomParamf specular_tint;
    specular_tint.name = "specular_tint";
    specular_tint.parameter = cuda_material.specular_tint;
    material.c_paramf.push_back(specular_tint);

    data::CustomParamf specular_trans;
    specular_trans.name = "specular_trans";
    specular_trans.parameter = cuda_material.specular_trans;
    material.c_paramf.push_back(specular_trans);

    data::CustomParamf sheen;
    sheen.name = "sheen";
    sheen.parameter = cuda_material.sheen;
    material.c_paramf.push_back(sheen);

    data::CustomParamf sheen_tint;
    sheen_tint.name = "sheen_tint";
    sheen_tint.parameter = cuda_material.sheen_tint;
    material.c_paramf.push_back(sheen_tint);

    data::CustomParamf roughness;
    roughness.name = "roughness";
    roughness.parameter = cuda_material.roughness;
    material.c_paramf.push_back(roughness);

    data::CustomParamf metallic;
    metallic.name = "metallic";
    metallic.parameter = cuda_material.metallic;
    material.c_paramf.push_back(metallic);

    data::CustomParamf clearcoat;
    clearcoat.name = "clearcoat";
    clearcoat.parameter = cuda_material.clearcoat;
    material.c_paramf.push_back(clearcoat);

    data::CustomParamf clearcoat_gloss;
    clearcoat_gloss.name = "clearcoat_gloss";
    clearcoat_gloss.parameter = cuda_material.clearcoat_gloss;
    material.c_paramf.push_back(clearcoat_gloss);

    data::CustomParamf subsurface;
    subsurface.name = "subsurface";
    subsurface.parameter = cuda_material.subsurface;
    material.c_paramf.push_back(subsurface);

    data::CustomParamf anisotropic;
    anisotropic.name = "anisotropic";
    anisotropic.parameter = cuda_material.anisotropic;
    material.c_paramf.push_back(anisotropic);
}

void CreateSphere(MeshData& data, float radius, uint32_t material_id)
{

    uint32_t stacks = 50;
    uint32_t slices = 50;

    uint32_t vertices = (stacks + 1) * (slices + 1);
    uint32_t indices = stacks * slices * 2;
    
    //動的配列の確保
    data.position.resize(vertices * 3);
    data.normal.resize(vertices * 3);
    data.uv.resize(vertices * 2);
    data.index.resize(indices * 3);
    data.material_ids.resize(indices, material_id);
    data.vertex_count = vertices;
    data.index_count = indices * 3;

    // 頂点の位置とテクスチャ座標を求める
    for (uint32_t k = 0, j = 0; j <= stacks; ++j)
    {
        const float t(static_cast<float>(j) / static_cast<float>(stacks));
        const float ph(3.141593f * t);
        const float y(cos(ph));
        const float r(sin(ph));

        for (uint32_t i = 0; i <= slices; ++i)
        {
            const float s(static_cast<float>(i) / static_cast<float>(slices));
            const float th(-2.0f * 3.141593f * s);
            const float x(r * cos(th));
            const float z(r * sin(th));

            // 頂点の座標値
            data.position.at(k * 3) = x * radius;
            data.position.at(k * 3 + 1) = y * radius;
            data.position.at(k * 3 + 2) = z * radius;

            // 頂点の法線ベクトル
            data.normal.at(k * 3) = x;
            data.normal.at(k * 3 + 1) = y;
            data.normal.at(k * 3 + 2) = z;

            // 頂点のテクスチャ座標値
            data.uv.at(k * 2) = s;
            data.uv.at(k * 2 + 1) = t;

            ++k;
        }
    }

    // 面の指標を求める
    for (uint32_t k = 0, j = 0; j < stacks; ++j)
    {
        for (uint32_t i = 0; i < slices; ++i)
        {
            const int count((slices + 1) * j + i);

            // 上半分の三角形
            data.index.at(k * 3) = count;
            data.index.at(k * 3 + 1) = count + slices + 2;
            data.index.at(k * 3 + 2) = count + 1;
            ++k;

            // 下半分の三角形
            data.index.at(k * 3) = count;
            data.index.at(k * 3 + 1) = count + slices + 1;
            data.index.at(k * 3 + 2) = count + slices + 2;
            ++k;
        }
    }

}
void GenerateData(data::Scene& object, SphereParam& param, uint32_t material_offset)
{
    data::Model model = {};
    model.mesh_id.push_back(0);
    model.name = param.name + "_model";
    model.transform_id = 0;
    

    data::Mesh mesh = {};
    mesh.name = param.name + "_mesh";
    mesh.index_buffer_id = 0;
    mesh.vertex_buffer_id[data::AttributeType::Position] = 1;
    mesh.vertex_buffer_id[data::AttributeType::Normal] = 2;
    mesh.vertex_buffer_id[data::AttributeType::TexCoord1] = 3;
    mesh.vertex_buffer_id[data::AttributeType::Material] = 4;
    mesh.material_id = material_offset;

    MeshData mesh_data = {};
    CreateSphere(mesh_data, param.radius, mesh.material_id);

    data::Buffer vertex_buffer;
    vertex_buffer.buffer_type = data::BufferType::VertexBuffer;
    vertex_buffer.value_type = data::ValueType::Float;
    vertex_buffer.element_type = data::ElementType::Vector3;
    vertex_buffer.count = mesh_data.vertex_count;
    vertex_buffer.data.resize(mesh_data.position.size() * sizeof(float));
    vertex_buffer.name = param.name + "_pvb";
    memcpy(vertex_buffer.data.data(), mesh_data.position.data(), vertex_buffer.data.size());

    data::Buffer normal_buffer;
    normal_buffer.buffer_type = data::BufferType::VertexBuffer;
    normal_buffer.value_type = data::ValueType::Float;
    normal_buffer.element_type = data::ElementType::Vector3;
    normal_buffer.count = mesh_data.vertex_count;
    normal_buffer.data.resize(mesh_data.normal.size() * sizeof(float));
    normal_buffer.name = param.name + "_nvb";
    memcpy(normal_buffer.data.data(), mesh_data.normal.data(), normal_buffer.data.size());

    data::Buffer uv_buffer;
    uv_buffer.buffer_type = data::BufferType::VertexBuffer;
    uv_buffer.value_type = data::ValueType::Float;
    uv_buffer.element_type = data::ElementType::Vector2;
    uv_buffer.name = param.name + "_tvb";
    uv_buffer.count = mesh_data.vertex_count;
    uv_buffer.data.resize(mesh_data.uv.size() * sizeof(float));
    memcpy(uv_buffer.data.data(), mesh_data.uv.data(), uv_buffer.data.size());

    data::Buffer material_buffer;
    material_buffer.buffer_type = data::BufferType::VertexBuffer;
    material_buffer.value_type = data::ValueType::Uint32;
    material_buffer.element_type = data::ElementType::Scalar;
    material_buffer.name = param.name + "_mvb";
    material_buffer.count = mesh_data.index_count / 3;
    material_buffer.data.resize(mesh_data.material_ids.size() * sizeof(uint32_t));
    memcpy(material_buffer.data.data(), mesh_data.material_ids.data(), material_buffer.data.size());

    data::Buffer index_buffer;
    index_buffer.buffer_type = data::BufferType::IndexBuffer;
    index_buffer.value_type = data::ValueType::Uint32;
    index_buffer.element_type = data::ElementType::Scalar;
    index_buffer.name = param.name + "_ib";
    index_buffer.count = mesh_data.index_count;
    index_buffer.data.resize(mesh_data.index.size() * sizeof(uint32_t));
    memcpy(index_buffer.data.data(), mesh_data.index.data(), index_buffer.data.size());

    data::Transform trs;
    trs.translate[0] = param.position[0];
    trs.translate[1] = param.position[1];
    trs.translate[2] = param.position[2];

    trs.rotation[0] = param.rotation[0];
    trs.rotation[1] = param.rotation[1];
    trs.rotation[2] = param.rotation[2];
    trs.rotation[3] = param.rotation[3];

    trs.scale[0] = param.scale[0];
    trs.scale[1] = param.scale[1];
    trs.scale[2] = param.scale[2];
    trs.is_matrix = false;


    data::Material material;
    material.name = param.name + "_material";
    ConvertMaterial(material, param.material);

    object.buffers.push_back(index_buffer);
    object.buffers.push_back(vertex_buffer);
    object.buffers.push_back(normal_buffer);
    object.buffers.push_back(uv_buffer);
    object.buffers.push_back(material_buffer);

    object.models.push_back(model);
    object.meshes.push_back(mesh);
    object.transforms.push_back(trs);
    object.materials.push_back(material);
}

int32_t GenerateSphereLight(GraphicsContext& context, std::string name, CudaLight& light, ResoucePool& resouce_pool)
{
    resouce_pool.light_table.push_back(light);
    data::Scene light_object = {};
    SphereParam light_object_param = {};
    light_object_param.name = name;
    light_object_param.radius = light.radius;
    light_object_param.position[0] = light.position.x;
    light_object_param.position[1] = light.position.y;
    light_object_param.position[2] = light.position.z;

    light_object_param.scale[0] = 1.0f;
    light_object_param.scale[1] = 1.0f;
    light_object_param.scale[2] = 1.0f;

    light_object_param.material.emission[0] = light.emission.x;
    light_object_param.material.emission[1] = light.emission.y;
    light_object_param.material.emission[2] = light.emission.z;

    uint32_t material_offset = (uint32_t)resouce_pool.material_table.size();
    GenerateData(light_object, light_object_param, material_offset);
    CreateObject(light_object, context, resouce_pool);
    return (int32_t)(resouce_pool.instance_as_table.size() - 1);
}

void GenerateBSDFSample(GraphicsContext& context, ResoucePool& resouce_pool)
{
    for (size_t i = 0; i < 90; i++)
    {
        SphereParam param;
        size_t y = i % 10;
        size_t x = i / 10;
        float t = static_cast<float>(y) / 10;
        if (x == 0)
        {
            param.material.base_color[0] = 1.0f;
            param.material.base_color[1] = 0.25f;
            param.material.base_color[2] = 0.0f;
            param.material.sheen= TMax(t, 0.00001f);
            param.material.specular_trans = 0.0f;
            param.material.metallic = 0.0f;
            param.material.clearcoat = 0.0f;
            param.material.roughness = 0.5f;
        }
        else if (x == 1)
        {
            param.material.base_color[0] = 1.0f;
            param.material.base_color[1] = 0.25f;
            param.material.base_color[2] = 0.0f;
            param.material.sheen_tint = t;
            param.material.sheen = 1.0f;
            param.material.specular_trans = 0.0f;
            param.material.metallic = 0.0f;
            param.material.clearcoat = 0.0f;
            param.material.roughness = 0.5f;
        }
        else if (x == 2)
        {
            param.material.base_color[0] = 1.0f;
            param.material.base_color[1] = 0.5f;
            param.material.base_color[2] = 0.5f;
            param.material.subsurface = t;
            param.material.sheen_tint = 0.0f;
            param.material.sheen = 0.0f;
            param.material.specular_trans = 0.0f;
            param.material.metallic = 0.0f;
            param.material.clearcoat = 0.0f;
            param.material.roughness = 0.5f;
        }
        else if (x == 3)
        {
            param.material.base_color[0] = 0.1f;
            param.material.base_color[1] = 0.1f;
            param.material.base_color[2] = 0.1f;
            param.material.roughness = TMax(t, 0.001f);
            param.material.metallic = 0.3f;
            param.material.specular_tint = 0.0f;
        }
        else if (x == 4)
        {
            param.material.base_color[0] = 1.0f;
            param.material.base_color[1] = 1.0f;
            param.material.base_color[2] = 0.0f;
            param.material.metallic = t;
            param.material.ior = 1.0f;
            param.material.specular_tint = 0.0f;
            param.material.anisotropic = 0.0f;
        }
        else if (x == 5)
        {
            param.material.base_color[0] = 0.5f;
            param.material.base_color[1] = 1.0f;
            param.material.base_color[2] = 0.5f;
            param.material.anisotropic = t;
            param.material.metallic = 0.8f;
            param.material.roughness = 0.3f;
            param.material.ior = 1.0f;
            param.material.specular_tint = 0.0f;
        }
        else if (x == 6)
        {
            param.material.base_color[0] = 0.0f;
            param.material.base_color[1] = 1.0f;
            param.material.base_color[2] = 0.0f;
            param.material.anisotropic = 0.0f;
            param.material.metallic = 0.3f;
            param.material.roughness = 0.3f;
            param.material.ior = 1.0f;
            param.material.specular_tint = t;
        }
        else if (x == 7)
        {
            param.material.base_color[0] = 0.5f;
            param.material.base_color[1] = 0.0f;
            param.material.base_color[2] = 0.5f;
            param.material.clearcoat = t;
        }
        else if (x == 8)
        {
            param.material.base_color[0] = 0.5f;
            param.material.base_color[1] = 0.5f;
            param.material.base_color[2] = 0.0f;
            param.material.clearcoat_gloss = t;
            param.material.clearcoat = 1.0f;
        }
#if 0
        else if (x == 9)
        {
            param.material.base_color[0] = 0.0f;
            param.material.base_color[1] = 0.0f;
            param.material.base_color[2] = 1.0f;
            param.material.specular_trans = t;
            param.material.ior = 1.51714;
        }
        else if (x == 10)
        {
            param.material.base_color[0] = 1.0f;
            param.material.base_color[1] = 1.0f;
            param.material.base_color[2] = 1.0f;
            param.material.specular_trans = 1.0f;
            param.material.ior = t * 5.0f + 1.0f;
        }
        else if (x == 11)
        {
            param.material.base_color[0] = 1.0f;
            param.material.base_color[1] = 0.0f;
            param.material.base_color[2] = 0.0f;
            param.material.specular_trans = 1.0f;
            param.material.relative_ior = t;
        }
#endif
        param.name = "bsdf_sample_" + std::to_string(i);
        param.radius = 1.0f;
        param.position[0] = y * 2.5f;
        param.position[1] = x * 2.5f;
        param.position[2] = 0.0f;
        param.scale[0] = 1.0f;
        param.scale[1] = 1.0f;
        param.scale[2] = 1.0f;
        param.rotation[0] = 0.0f;
        param.rotation[1] = 0.0f;
        param.rotation[2] = 0.0f;
        param.rotation[3] = 1.0f;

        data::Scene object;
        uint32_t material_offset = 0;
        if (resouce_pool.material_table.size() > 0)
        {
            material_offset = (uint32_t)resouce_pool.material_table.size();
        }
        GenerateData(object, param, material_offset);
        CreateObject(object, context, resouce_pool);
    }
}

void ParticleSystem::GenerarteParticleLight(GraphicsContext& context, ResoucePool& resouce_pool, ParticleSystemParam& param)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    m_param = param;

    for (uint32_t  i = 0; i < m_param.particle_num; i++) 
    {
        CudaLight light = {};
        std::string name = param.name + std::to_string(i);
        light.emission.x = param.emission[0];
        light.emission.y = param.emission[1];
        light.emission.z = param.emission[2];
        light.radius = param.radius;
        light.type = LightType::Sphere;
        light.position.x = ((float)mt() / (float)mt.max() - 0.5f) * 2.0f * param.range + param.center[0];
        light.position.y = param.center[1];
        light.position.z = ((float)mt() / (float)mt.max() - 0.5f) * 2.0f * param.range + param.center[2];
        GenerateSphereLight(context, name, light, resouce_pool);

        uint32_t instance_id = (uint32_t)resouce_pool.instance_as_table.size() - 1;
        uint32_t light_id = (uint32_t)resouce_pool.light_table.size() - 1;
        uint32_t material_id = (uint32_t)resouce_pool.material_table.size() - 1;

        float3 velocity = {};
        velocity.x = ((float)mt() / (float)mt.max() - 0.5f) * 2.0f;
        velocity.y = ((float)mt() / (float)mt.max() - 0.5f) * 2.0f;
        velocity.z = ((float)mt() / (float)mt.max() - 0.5f) * 2.0f;

        m_instance_ids.push_back(instance_id);
        m_light_ids.push_back(light_id);
        m_material_ids.push_back(material_id);

        m_host_position.push_back(light.position);
        m_host_scale.push_back(1.0f);
        m_host_velocity.push_back(velocity);
        m_host_emission.push_back(light.emission);
    }

    m_position[0].CreateBuffer(context, param.particle_num * sizeof(float3), data::ValueType::Float, data::ElementType::Vector3, param.name + "_pos0", m_host_position.data());
    m_position[1].CreateBuffer(context, param.particle_num * sizeof(float3), data::ValueType::Float, data::ElementType::Vector3, param.name + "_pos1", m_host_position.data());

    m_velocity[0].CreateBuffer(context, param.particle_num * sizeof(float3), data::ValueType::Float, data::ElementType::Vector3, param.name + "_vel0", m_host_velocity.data());
    m_velocity[1].CreateBuffer(context, param.particle_num * sizeof(float3), data::ValueType::Float, data::ElementType::Vector3, param.name + "_vel1", m_host_velocity.data());

    m_emission[0].CreateBuffer(context, param.particle_num * sizeof(float3), data::ValueType::Float, data::ElementType::Vector3, param.name + "_emi0", m_host_emission.data());
    m_emission[1].CreateBuffer(context, param.particle_num * sizeof(float3), data::ValueType::Float, data::ElementType::Vector3, param.name + "_emi1", m_host_emission.data());

    m_scale[0].CreateBuffer(context, param.particle_num * sizeof(float), data::ValueType::Float, data::ElementType::Scalar, param.name + "_sca0", m_host_scale.data());
    m_scale[1].CreateBuffer(context, param.particle_num * sizeof(float), data::ValueType::Float, data::ElementType::Scalar, param.name + "_sca1", m_host_scale.data());

    initialize = true;
}

void ParticleSystem::UpdateParticleLight(GraphicsContext& context, ResoucePool& resouce_pool, CUfunction& cuda_particle, uint32_t framecount)
{
    if (!initialize) 
    {
        return;
    }
    int numElements = m_param.particle_num;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    dim3 cudaBlockSize(threadsPerBlock, 1, 1);
    dim3 cudaGridSize(blocksPerGrid, 1, 1);

    uint32_t cur_buffer_index = (m_buffer_index + 1) % 2;
    uint32_t pre_buffer_index = m_buffer_index;

    void* arr[] =
    {
        reinterpret_cast<void*>(&m_position[cur_buffer_index].buffer),
        reinterpret_cast<void*>(&m_position[pre_buffer_index].buffer),
        reinterpret_cast<void*>(&m_scale[cur_buffer_index].buffer),
        reinterpret_cast<void*>(&m_scale[pre_buffer_index].buffer),
        reinterpret_cast<void*>(&m_velocity[cur_buffer_index].buffer),
        reinterpret_cast<void*>(&m_velocity[pre_buffer_index].buffer),
        reinterpret_cast<void*>(&framecount),
        reinterpret_cast<void*>(&numElements),
    };

    CUDA_CHECK_ERROR(cuLaunchKernel(cuda_particle, cudaGridSize.x, cudaGridSize.y, cudaGridSize.z,
        cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, 0, nullptr, arr, 0));

    CUDA_CHECK_ERROR(cuCtxSynchronize());

    CUDA_CHECK(cudaMemcpy(m_host_position.data(), reinterpret_cast<void*>(m_position[cur_buffer_index].buffer), m_position[cur_buffer_index].byte_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(m_host_scale.data(), reinterpret_cast<void*>(m_scale[cur_buffer_index].buffer), m_scale[cur_buffer_index].byte_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(m_host_emission.data(), reinterpret_cast<void*>(m_emission[cur_buffer_index].buffer), m_emission[cur_buffer_index].byte_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < m_param.particle_num; i++) 
    {
        uint32_t instance_id = m_instance_ids.at(i);
        auto& instanceAS = resouce_pool.instance_as_table.at(instance_id);

        instanceAS.transform.translate.x = m_host_position.at(i).x;
        instanceAS.transform.translate.y = m_host_position.at(i).y;
        instanceAS.transform.translate.z = m_host_position.at(i).z;
        /*
        instanceAS.transform.scale.x = m_host_scale.at(i);
        instanceAS.transform.scale.y = m_host_scale.at(i);
        instanceAS.transform.scale.z = m_host_scale.at(i);
        */
        instanceAS.transform.update();

        uint32_t light_index = m_light_ids.at(i);
        auto& light = resouce_pool.light_table.at(light_index);
        light.position = m_host_position.at(i);
        /*
* 
        light.radius = m_host_scale.at(i);
        light.emission = m_host_emission.at(i);
        */
    }

    m_buffer_index++;
    m_buffer_index %= 2;
}
void GenerateCornelBoxSample(GraphicsContext& context, ResoucePool& resouce_pool) 
{
    auto& vertices  = cornel_box::g_vertices;

    std::vector<float3> positons;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    std::vector<uint32_t> indices;
    for (int i = 0; i < vertices.size(); i++) 
    {
        float3 position;
        position.x = vertices.at(i).x;
        position.y = vertices.at(i).y;
        position.z = vertices.at(i).z;
        
        float3 normal;
        normal.x = 0.0f;
        normal.y = 1.0f;
        normal.z = 0.0f;

        float2 uv;
        uv.x = 0.0f;
        uv.y = 0.0f;
        uvs.push_back(uv);
        positons.push_back(position);
        indices.push_back(i);
        normals.push_back(normal);
        
    }

    std::vector<int32_t> materials;
    for (int i = 0; i < cornel_box::g_mat_indices.size(); i++) 
    {
        materials.push_back(cornel_box::g_mat_indices.at(i));
    }


    data::Model model = {};
    model.mesh_id.push_back(0);
    model.name =  "cornel_model";
    model.transform_id = 0;


    data::Mesh mesh = {};
    mesh.name = "cornel_mesh";
    mesh.index_buffer_id = 0;
    mesh.vertex_buffer_id[data::AttributeType::Position] = 1;
    mesh.vertex_buffer_id[data::AttributeType::Normal] = 2;
    mesh.vertex_buffer_id[data::AttributeType::TexCoord1] = 3;
    mesh.vertex_buffer_id[data::AttributeType::Material] = 4;
    mesh.material_id = 0;

    data::Buffer vertex_buffer;
    vertex_buffer.buffer_type = data::BufferType::VertexBuffer;
    vertex_buffer.value_type = data::ValueType::Float;
    vertex_buffer.element_type = data::ElementType::Vector3;
    vertex_buffer.count = (uint32_t)positons.size();
    vertex_buffer.data.resize(positons.size() * sizeof(float3));
    vertex_buffer.name = "cornelbox_pvb";
    memcpy(vertex_buffer.data.data(), positons.data(), vertex_buffer.data.size());

    data::Buffer normal_buffer;
    normal_buffer.buffer_type = data::BufferType::VertexBuffer;
    normal_buffer.value_type = data::ValueType::Float;
    normal_buffer.element_type = data::ElementType::Vector3;
    normal_buffer.count = (uint32_t)normals.size();
    normal_buffer.data.resize(normals.size() * sizeof(float3));
    normal_buffer.name = "cornelbox_nvb";
    memcpy(normal_buffer.data.data(), normals.data(), normal_buffer.data.size());

    data::Buffer uv_buffer;
    uv_buffer.buffer_type = data::BufferType::VertexBuffer;
    uv_buffer.value_type = data::ValueType::Float;
    uv_buffer.element_type = data::ElementType::Vector2;
    uv_buffer.name = "cornelbox_tvb";
    uv_buffer.count = (uint32_t)uvs.size();
    uv_buffer.data.resize(uvs.size() * sizeof(float2));
    memcpy(uv_buffer.data.data(), uvs.data(), uv_buffer.data.size());

    data::Buffer material_buffer;
    material_buffer.buffer_type = data::BufferType::VertexBuffer;
    material_buffer.value_type = data::ValueType::Uint32;
    material_buffer.element_type = data::ElementType::Scalar;
    material_buffer.name = "material_mvb";
    material_buffer.count = (uint32_t)materials.size();
    material_buffer.data.resize(materials.size() * sizeof(uint32_t));
    memcpy(material_buffer.data.data(), materials.data(), material_buffer.data.size());

    data::Buffer index_buffer;
    index_buffer.buffer_type = data::BufferType::IndexBuffer;
    index_buffer.value_type = data::ValueType::Uint32;
    index_buffer.element_type = data::ElementType::Scalar;
    index_buffer.name = "cornelbox_ib";
    index_buffer.count = (uint32_t)indices.size();
    index_buffer.data.resize(indices.size() * sizeof(uint32_t));
    memcpy(index_buffer.data.data(), indices.data(), index_buffer.data.size());

    data::Transform trs;
    trs.translate[0] = 0.0f;
    trs.translate[1] = 0.0f;
    trs.translate[2] = 0.0f;

    trs.rotation[0] = 0.0f;
    trs.rotation[1] = 0.0f;
    trs.rotation[2] = 0.0f;
    trs.rotation[3] = 1.0f;

    trs.scale[0] = 1.0f;
    trs.scale[1] = 1.0f;
    trs.scale[2] = 1.0f;
    trs.is_matrix = false;

    data::Scene object;
    for (int32_t i = 0; i < 4; i++)
    {

        CudaMaterial cudaMaterial = {};
        cudaMaterial.base_color[0] = cornel_box::g_diffuse_colors[i].x;
        cudaMaterial.base_color[1] = cornel_box::g_diffuse_colors[i].y;
        cudaMaterial.base_color[2] = cornel_box::g_diffuse_colors[i].z;
        if (i == 0)
        {
            cudaMaterial.roughness = 0.5f;
            cudaMaterial.metallic = 1.0f;
        }
        else if (i == 1) 
        {
            cudaMaterial.roughness = 0.0f;
            cudaMaterial.metallic = 0.0f;
        }
        else if (i == 2)
        {
            cudaMaterial.roughness = 0.5f;
            cudaMaterial.metallic = 0.5f;
        }
        else if (i == 3)
        {
            cudaMaterial.roughness = 1.0f;
            cudaMaterial.metallic = 0.0f;
        }
        data::Material material;
        material.name = "cornelbox_material" + std::to_string(i);
       ConvertMaterial(material, cudaMaterial);
       object.materials.push_back(material);

    }

    object.buffers.push_back(index_buffer);
    object.buffers.push_back(vertex_buffer);
    object.buffers.push_back(normal_buffer);
    object.buffers.push_back(uv_buffer);
    object.buffers.push_back(material_buffer);

    object.models.push_back(model);
    object.meshes.push_back(mesh);
    object.transforms.push_back(trs);

    CreateObject(object, context, resouce_pool);
  
}
}