/**
 * @file    GpuResource.cpp
 * @brief   GPUリソースのソースファイル
 */

#pragma once

#include "graphic/GpuResouce.hpp"
#include "utility/SystemPath.hpp"
#include "utility/FileSystem.hpp"
#include "utility/ImageLoader.hpp"

namespace slug 
{
size_t ConvertValueTypeBitSize(data::ValueType value_type)
{
    switch (value_type) {
    case data::ValueType::Uint8:
        return sizeof(uint8_t);
        break;
    case data::ValueType::Sint8:
        return sizeof(int8_t);
        break;
    case data::ValueType::Uint16:
        return sizeof(uint16_t);
        break;
    case data::ValueType::Sint16:
        return sizeof(int16_t);
        break;
    case data::ValueType::Uint32:
        return sizeof(uint32_t);
        break;
    case data::ValueType::Sint32:
        return sizeof(int32_t);
        break;
    case data::ValueType::Float:
        return sizeof(float);
        break;
    case data::ValueType::Double:
        return sizeof(double);
        break;
    case data::ValueType::LDouble:
        return sizeof(long double);
        break;
    default:
        return sizeof(uint32_t);
    }
}

size_t ConvertElementTypeBitSize(data::ElementType element_type)
{
    switch (element_type) {
    case data::ElementType::Vector1:
        return 1;
        break;
    case data::ElementType::Vector2:
        return 2;
        break;
    case data::ElementType::Vector3:
        return 3;
        break;
    case data::ElementType::Vector4:
        return 4;
        break;
    case data::ElementType::Matrix2x2:
        return 4;
        break;
    case data::ElementType::Matrix3x3:
        return 9;
        break;
    case data::ElementType::Matrix4x4:
        return 16;
        break;
    case data::ElementType::Scalar:
        return 1;
        break;
    case data::ElementType::Vector:
        return 4;
        break;
    case data::ElementType::Matrix:
        return 16;
        break;
    default:
        return 1;
        break;
    }
}

OptixVertexFormat ConvertOptixVertexFormat(data::ValueType value_type, data::ElementType element_type)
{
    if (element_type == data::ElementType::Vector3 || element_type == data::ElementType::Vector4)
    {
        if (value_type == data::ValueType::Float)
        {
            return OPTIX_VERTEX_FORMAT_FLOAT3;
        }
    }
    else if (element_type == data::ElementType::Vector2)
    {
        if (value_type == data::ValueType::Float)
        {
            return OPTIX_VERTEX_FORMAT_FLOAT2;
        }
    }
    return OPTIX_VERTEX_FORMAT_NONE;
}

OptixIndicesFormat ConvertIndincesFormat(data::ValueType value_type)
{
    if (value_type == data::ValueType::Uint32 || value_type == data::ValueType::Sint32)
    {
        return OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    }
    else if (value_type == data::ValueType::Uint16 || value_type == data::ValueType::Sint16)
    {
        return OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
    }
    return OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
}


size_t CalcBufferBitSize(data::ValueType value_type, data::ElementType element_type, uint32_t count)
{
    size_t value_size = ConvertValueTypeBitSize(value_type);
    size_t element_size = ConvertElementTypeBitSize(element_type);
    return value_size * element_size * count;
}

size_t CalcBufferStride(data::ValueType value_type, data::ElementType element_type)
{
    size_t value_size = ConvertValueTypeBitSize(value_type);
    size_t element_size = ConvertElementTypeBitSize(element_type);
    return value_size * element_size;
}

bool ConvertAttribute(data::AttributeType type, VertexAttribute& value)
{

    if (type == data::AttributeType::Position) {
        value = VertexAttribute::Position;
        return true;
    }
    else if (type == data::AttributeType::TexCoord1) {
        value = VertexAttribute::Texcoord;
        return true;
    }
    else if (type == data::AttributeType::Normal) {
        value = VertexAttribute::Normal;
        return true;
    }
    else if (type == data::AttributeType::Material) {
        value = VertexAttribute::MaterialIndex;
        return true;
    }
    return false;
}

void UpdateTransform(InstanceAS& instanceAS)
{
    instanceAS.instance.transform[0] = instanceAS.transform.world[0][0];
    instanceAS.instance.transform[1] = instanceAS.transform.world[1][0];
    instanceAS.instance.transform[2] = instanceAS.transform.world[2][0];
    instanceAS.instance.transform[3] = instanceAS.transform.world[3][0];

    instanceAS.instance.transform[4] = instanceAS.transform.world[0][1];
    instanceAS.instance.transform[5] = instanceAS.transform.world[1][1];
    instanceAS.instance.transform[6] = instanceAS.transform.world[2][1];
    instanceAS.instance.transform[7] = instanceAS.transform.world[3][1];

    instanceAS.instance.transform[8] = instanceAS.transform.world[0][2];
    instanceAS.instance.transform[9] = instanceAS.transform.world[1][2];
    instanceAS.instance.transform[10] = instanceAS.transform.world[2][2];
    instanceAS.instance.transform[11] = instanceAS.transform.world[3][2];
}

void CreateGeometryResource(GraphicsContext& context, data::Scene& object, data::Mesh& mesh, data::Model& model, ResoucePool& resouce_pool, uint32_t mesh_id)
{
    // インデックス情報
    uint32_t indices = 0;
    uint32_t index_buffer_id = {};
    if (object.buffers.size() > mesh.index_buffer_id)
    {
        data::Buffer& index_buffer = object.buffers.at(mesh.index_buffer_id);
        uint64_t index_buffer_size = CalcBufferBitSize(index_buffer.value_type, index_buffer.element_type, index_buffer.count);
        if (!resouce_pool.Add(resouce_pool.buffer_table, index_buffer.name, index_buffer_id))
        {
            auto& cuda_buffer = resouce_pool.buffer_table.at(index_buffer_id);
            if (cuda_buffer.buffer == 0)
            {
                context.CreateBuffer(cuda_buffer.buffer, index_buffer_size, index_buffer.data.data());
                cuda_buffer.attribute = VertexAttribute::Index;
                cuda_buffer.element_type = index_buffer.element_type;
                cuda_buffer.value_type = index_buffer.value_type;
                cuda_buffer.name = index_buffer.name;
                cuda_buffer.byte_size = index_buffer_size;
            }
            indices = (uint32_t)index_buffer.count;
        }
    }

    // 頂点情報
    uint32_t vertices = 0;
    uint32_t position_buffer_id = 0;
    uint32_t normal_buffer_id = 0;
    uint32_t texcoord_buffer_id = 0;
    uint32_t material_buffer_id = 0;
    bool is_found_material_buffer = false;
    {
        uint32_t vertex_buffer_id = 0;
        for (auto& vertex_buffer_attributes : mesh.vertex_buffer_id)
        {
            if (object.buffers.size() > vertex_buffer_attributes.second)
            {
                data::Buffer& vertex_buffer = object.buffers.at(vertex_buffer_attributes.second);
                std::string vertex_buffer_name = object.name + "_" + vertex_buffer.name +"_vb";
                uint64_t vertex_buffer_size = (uint64_t)CalcBufferBitSize(vertex_buffer.value_type, vertex_buffer.element_type, vertex_buffer.count);
                VertexAttribute attribute = {};
                if (ConvertAttribute(vertex_buffer_attributes.first, attribute))
                {
                    if (!resouce_pool.Add(resouce_pool.buffer_table, vertex_buffer_name, vertex_buffer_id))
                    {
                        auto& cuda_buffer = resouce_pool.buffer_table.at(vertex_buffer_id);
                        if (cuda_buffer.buffer == 0)
                        {
                            context.CreateBuffer(cuda_buffer.buffer, vertex_buffer_size, vertex_buffer.data.data());
                            cuda_buffer.attribute = attribute;
                            cuda_buffer.element_type = vertex_buffer.element_type;
                            cuda_buffer.value_type = vertex_buffer.value_type;
                            cuda_buffer.name = vertex_buffer_name;
                            cuda_buffer.byte_size = vertex_buffer_size;
                        }
                    }
                }

                vertices = TMax(vertex_buffer.count, vertices);
                switch (attribute)
                {
                case VertexAttribute::Position:
                    position_buffer_id = vertex_buffer_id;
                    break;
                case VertexAttribute::Normal:
                    normal_buffer_id = vertex_buffer_id;
                    break;
                case VertexAttribute::Texcoord:
                    texcoord_buffer_id = vertex_buffer_id;
                    break;
                case VertexAttribute::MaterialIndex:
                    material_buffer_id = vertex_buffer_id;
                    is_found_material_buffer = true;
                    break;
                default:
                    break;
                }
                size_t offset_name = vertex_buffer.name.find_last_of("_vb") + 1;
                vertex_buffer_name += vertex_buffer.name.substr(offset_name, vertex_buffer.name.size() - offset_name).c_str();
            }
        }
    }

    if (!is_found_material_buffer)
    {
        uint32_t vertex_buffer_id = 0;
        std::string vertex_buffer_name = object.name + "_" + mesh.name + "_material_id" + "_vb";
        if (!resouce_pool.Add(resouce_pool.buffer_table, vertex_buffer_name, vertex_buffer_id))
        {
            auto& cuda_buffer = resouce_pool.buffer_table.at(vertex_buffer_id);
            if (cuda_buffer.buffer == 0)
            {
                data::Buffer& index_buffer = object.buffers.at(mesh.index_buffer_id);
                size_t vertex_buffer_size = index_buffer.count / 3 * sizeof(int32_t);
                std::vector<int32_t> material_data = {};
                material_data.resize(index_buffer.count / 3, mesh.material_id);
                context.CreateBuffer(cuda_buffer.buffer, vertex_buffer_size, material_data.data());
                cuda_buffer.attribute = VertexAttribute::MaterialIndex;
                cuda_buffer.element_type = data::ElementType::Scalar;
                cuda_buffer.value_type = data::ValueType::Sint32;
                cuda_buffer.name = vertex_buffer_name;
                cuda_buffer.byte_size = vertex_buffer_size;
            }
        }
        material_buffer_id = vertex_buffer_id;
    }

    // ジオメトリ加速構造体
    uint32_t geometry_as_id = {};
    std::string geometry_as_name = object.name + "_" + model.name + "_" + mesh.name + std::to_string(mesh_id) + "_gas";
    if (!resouce_pool.Add(resouce_pool.geometry_as_table, geometry_as_name, geometry_as_id))
    {
        uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        build_input.triangleArray.flags = triangle_input_flags;
        build_input.triangleArray.numSbtRecords = 1;

        auto& vertex_buffer = resouce_pool.buffer_table.at(position_buffer_id);
        build_input.triangleArray.vertexBuffers = &vertex_buffer.buffer;
        build_input.triangleArray.vertexFormat = ConvertOptixVertexFormat(vertex_buffer.value_type, vertex_buffer.element_type);
        build_input.triangleArray.vertexStrideInBytes = (uint32_t)CalcBufferStride(vertex_buffer.value_type, vertex_buffer.element_type);
        build_input.triangleArray.numVertices = vertices;

        auto& index_buffer = resouce_pool.buffer_table.at(index_buffer_id);
        build_input.triangleArray.indexBuffer = index_buffer.buffer;
        build_input.triangleArray.indexFormat = ConvertIndincesFormat(index_buffer.value_type);
        build_input.triangleArray.indexStrideInBytes = (uint32_t)CalcBufferStride(index_buffer.value_type, index_buffer.element_type) * 3;
        build_input.triangleArray.numIndexTriplets = indices / 3;

        auto& geomeryAS = resouce_pool.geometry_as_table.at(geometry_as_id);
        geomeryAS.vertex_buffer_id = position_buffer_id;
        geomeryAS.index_buffer_id = index_buffer_id;
        geomeryAS.offset_buffer_id = 0;
        geomeryAS.normal_buffer_id = normal_buffer_id;
        geomeryAS.texcoord_buffer_id = texcoord_buffer_id;
        geomeryAS.material_buffer_id = material_buffer_id;
        geomeryAS.name = geometry_as_name;
        context.CreateAccelStructureTriangle(build_input, geomeryAS.handle, geomeryAS.output_buffer);
    }

    // transform情報
    uint32_t instance_as_id = 0;
    std::string instance_as_name = object.name + "_" + model.name + "_" + mesh.name + std::to_string(mesh_id) + "_ias";
    if (!resouce_pool.Add(resouce_pool.instance_as_table, instance_as_name, instance_as_id))
    {
        
        auto& transform = object.transforms.at(model.transform_id);
        Transform trs = {};
        trs.translate = { transform.translate[0], transform.translate[1], transform.translate[2] };
        trs.rotation = { transform.rotation[0], transform.rotation[1], transform.rotation[2] , transform.rotation[3]};
        trs.scale = { transform.scale[0], transform.scale[1], transform.scale[2] };
        trs.update();

        auto& instanceAS = resouce_pool.instance_as_table.at(instance_as_id);
        instanceAS.bind_geometry_as_id = geometry_as_id;
        instanceAS.instance.visibilityMask = 255;
        instanceAS.instance.instanceId = instance_as_id;
        instanceAS.instance.sbtOffset = instance_as_id * 2;
        instanceAS.name = instance_as_name;
        instanceAS.transform = trs;
        instanceAS.instance.traversableHandle = resouce_pool.geometry_as_table.at(geometry_as_id).handle;

        UpdateTransform(instanceAS);
    }

    resouce_pool.geometry_as_table.at(geometry_as_id).bind_instance_as_id = instance_as_id;
}

bool CreateModelResouce(GraphicsContext& context, data::Scene& object, data::Model& model, std::string parent_as_name, ResoucePool& resouce_pool)
{
    std::vector<uint32_t> instance_ids = {};
    for (size_t i = 0; i < model.mesh_id.size(); i++) {

        data::Mesh& mesh = object.meshes.at(model.mesh_id.at(i));
        CreateGeometryResource(context, object, mesh, model, resouce_pool, static_cast<uint32_t>(i));
    }
    return true;
}

void CreateMaterialResource(data::Scene& object, ResoucePool& resouce_pool)
{

    for (size_t i = 0; i < object.materials.size(); i++)
    {
        CudaMaterial cuda_material = {};
        auto& host_material = object.materials.at(i);
        for (auto itr : host_material.c_param4f)
        {
            if (itr.name == "diffuse")
            {
                cuda_material.base_color[0] = itr.parameter[0];
                cuda_material.base_color[1] = itr.parameter[1];
                cuda_material.base_color[2] = itr.parameter[2];
            }
            else if (itr.name == "base_color")
            {
                cuda_material.base_color[0] = itr.parameter[0];
                cuda_material.base_color[1] = itr.parameter[1];
                cuda_material.base_color[2] = itr.parameter[2];
            }
            else if (itr.name == "ambient")
            {
                // Not Implement
            }
            else if (itr.name == "specular")
            {
                // Not Implement
            }
            else if (itr.name == "transmittance")
            {
                cuda_material.specular_trans = 1.0f - itr.parameter[0];
            }
            else if (itr.name == "emission")
            {
                cuda_material.emission[0] = itr.parameter[0];
                cuda_material.emission[1] = itr.parameter[1];
                cuda_material.emission[2] = itr.parameter[2];

            }
        }

        for (auto itr : host_material.c_paramf)
        {
            if (itr.name == "shininess")
            {
                // Not Implement
            }
            else if (itr.name == "ior")
            {
                cuda_material.ior = itr.parameter;
            }
            else if (itr.name == "relative_ior")
            {
                cuda_material.relative_ior = itr.parameter;
            }
            else if (itr.name == "specular_tint")
            {
                cuda_material.specular_tint = itr.parameter;
            }
            else if (itr.name == "specular_trans")
            {
                cuda_material.specular_trans = itr.parameter;
            }
            else if (itr.name == "sheen_tint")
            {
                cuda_material.sheen_tint = itr.parameter;
            }
            else if (itr.name == "sheen")
            {
                cuda_material.sheen = itr.parameter;
            }
            else if (itr.name == "roughness")
            {
                cuda_material.roughness = itr.parameter;
            }
            else if (itr.name == "metallic")
            {
                cuda_material.metallic = itr.parameter;
            }
            else if (itr.name == "metalness")
            {
                cuda_material.metallic = itr.parameter;
            }
            else if (itr.name == "metallic")
            {
                cuda_material.metallic = itr.parameter;
            }
            else if (itr.name == "clearcoat")
            {
                cuda_material.clearcoat = itr.parameter;
            }
            else if (itr.name == "clearcoat_thickness")
            {
                cuda_material.clearcoat = itr.parameter;
            }
            else if (itr.name == "clearcoat_gloss")
            {
                cuda_material.clearcoat_gloss = itr.parameter;
            }
            else if (itr.name == "clearcoat_roughness")
            {
                cuda_material.clearcoat_gloss = itr.parameter;
            }
            else if (itr.name == "subsurface")
            {
                cuda_material.subsurface = itr.parameter;
            }
            else if (itr.name == "anisotropic")
            {
                cuda_material.anisotropic = itr.parameter;

            }
            else if (itr.name == "anisotropy")
            {
                cuda_material.anisotropic = itr.parameter;

            }
            else if (itr.name == "anisotropy_rotation")
            {
                // Not Implement
            }
        }

        for (auto itr : host_material.texture_ids) 
        {
            if (itr < resouce_pool.texture_table.size())
            {
                auto& tex = resouce_pool.texture_table.at(itr);
                if (tex.name.find("bump") != std::string::npos)
                {
                    cuda_material.bump = tex.texObj;
                }
                else 
                {
                    cuda_material.albedo = tex.texObj;
                }
            }
        }
        resouce_pool.material_table.push_back(cuda_material);
    }
}

void CreateTextureResouce(data::Scene& object, GraphicsContext& context, ResoucePool& resouce_pool)
{
    resouce_pool.texture_table.resize(object.textures.size());
    for (size_t i = 0; i < object.textures.size(); i++)
    {
        auto& cuda_texture = resouce_pool.texture_table.at(i);
        auto& host_texture = object.textures.at(i);

        std::string image_path = host_texture.path;
        ImageInfo image_info = {};
        LoadImageFile(image_path.c_str(), image_info);

        auto& data_layout = image_info.data_layout.at(0).at(0);
        context.CreateTextureArray(cuda_texture.texArray, image_info.width, image_info.height, image_info.cuda_channel_desc, static_cast<const char*>(image_info.internal_data->data()) + data_layout.data_offset);

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.addressMode[2] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = image_info.cuda_read_mode;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = true;// image_info.is_srgb;
        context.CreateTextureArrayObject(cuda_texture.texObj, cuda_texture.texArray, tex_desc, nullptr);
        cuda_texture.name = image_info.filename;
    }
}

void CreateEnvironmentResouce(GraphicsContext& context, ResoucePool& resouce_pool)
{
    auto& cuda_texture = resouce_pool.envrionment_texture;

    std::string image_path = GetDirectoryWithPackage() + "\\image\\" + "env_map_test.jpg";
    ImageInfo image_info = {};
    LoadImageFile(image_path.c_str(), image_info);

    if (image_info.is_cube)
    {
        auto& data_layout = image_info.data_layout.at(0).at(0);
        context.CreateTextureCubeArray(cuda_texture.texArray, image_info.width, image_info.height, image_info.cuda_channel_desc, static_cast<const char*>(image_info.internal_data->data()) + data_layout.data_offset);
    }
    else
    {
        // パノラマ画像の場合こちら
        auto& data_layout = image_info.data_layout.at(0).at(0);
        context.CreateTextureArray(cuda_texture.texArray, image_info.width, image_info.height, image_info.cuda_channel_desc, static_cast<const char*>(image_info.internal_data->data()) + data_layout.data_offset);
    }
    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.addressMode[2] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = image_info.cuda_read_mode;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = image_info.is_srgb;

    cudaResourceViewDesc view_desc = {};
    view_desc.width = image_info.width;
    view_desc.height = image_info.height;
    view_desc.depth = image_info.array_size;
    view_desc.format = image_info.cuda_format;
    view_desc.firstLayer = 0;
    view_desc.lastLayer = 0;
    view_desc.firstMipmapLevel = 0;
    view_desc.lastMipmapLevel = 1;

    context.CreateTextureArrayObject(cuda_texture.texObj, cuda_texture.texArray, tex_desc, nullptr);
    cuda_texture.name = image_info.filename;
}

void CreateObject(data::Scene& object, GraphicsContext& context, ResoucePool& resouce_pool)
{
    if (object.models.size() > 0)
    {
        CreateModelResouce(context, object, object.models.at(0), "root", resouce_pool);
    }

    CreateTextureResouce(object, context, resouce_pool);
    CreateMaterialResource(object, resouce_pool);
}

void CreatRootInstance(GraphicsContext& context, ResoucePool& resouce_pool, CUdeviceptr& d_temp_buffer, CUdeviceptr& d_comapct_tmp_buffer)
{
    std::string instance_as_group_name = "root_ias_group";
    std::vector<OptixInstance> instance_children = {};
    for (size_t i = 0; i<  resouce_pool.instance_as_table.size(); i++)
    {
        instance_children.push_back(resouce_pool.instance_as_table.at(i).instance);
    }

    context.CreateInstance(resouce_pool.root_instance_as.output_buffer, instance_children.data(), sizeof(OptixInstance) * instance_children.size());

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = resouce_pool.root_instance_as.output_buffer;
    build_input.instanceArray.numInstances = (uint32_t)instance_children.size();
    context.CreateAccelStructureInstance(build_input, resouce_pool.root_instance_as.handle, d_temp_buffer, d_comapct_tmp_buffer);
    
}

void UpdateRootInstance(GraphicsContext& context, ResoucePool& resouce_pool, CUstream& cu_stream, CUdeviceptr& d_temp_buffer, CUdeviceptr& d_comapct_tmp_buffer)
{
    std::vector<OptixInstance> instance_children = {};
    for (size_t i = 0; i < resouce_pool.instance_as_table.size(); i++)
    {
        instance_children.push_back(resouce_pool.instance_as_table.at(i).instance);
    }
    context.UpdateInstance(resouce_pool.root_instance_as.output_buffer, instance_children.data(), sizeof(OptixInstance) * instance_children.size());
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = resouce_pool.root_instance_as.output_buffer;
    build_input.instanceArray.numInstances = (uint32_t)instance_children.size();
    context.UpdateAccelStructureInstance(build_input, resouce_pool.root_instance_as.handle, cu_stream, d_temp_buffer, d_comapct_tmp_buffer);
}

} // namespace slug