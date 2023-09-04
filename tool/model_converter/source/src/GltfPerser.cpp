/**
 * @file    GltfPerser.cpp
 * @brief   Gltfのパースを行うクラスのソース
 *
 */

#include <tiny_gltf.h>
#include "GltfPerser.hpp"
#include "Utility.hpp"

/**
 * @brief サンプラー値
*/
enum GLTFSamplerValueType {
    GLTFSamplerValueType_Wrap = 0,
    GLTFSamplerValueType_Filter,
    GLTFSamplerValueType_Max
};

ValueType ConvertGLTFValueType(int32_t value_type)
{
    switch (value_type) {
    case TINYGLTF_COMPONENT_TYPE_BYTE:
        return ValueType::Sint8;
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return ValueType::Uint8;
        break;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
        return ValueType::Sint16;
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return ValueType::Uint16;
        break;
    case TINYGLTF_COMPONENT_TYPE_INT:
        return ValueType::Sint32;
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        return ValueType::Uint32;
        break;
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        return ValueType::Float;
        break;
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
        return ValueType::Float;
        break;
    default:
        return ValueType::Uint32;
    }
}

ElementType ConvertGLTFElementType(int32_t element_type)
{
    switch (element_type) {
    case TINYGLTF_TYPE_VEC2:
        return ElementType::Vector2;
        break;
    case TINYGLTF_TYPE_VEC3:
        return ElementType::Vector3;
        break;
    case TINYGLTF_TYPE_VEC4:
        return ElementType::Vector4;
        break;
    case TINYGLTF_TYPE_MAT2:
        return ElementType::Matrix2x2;
        break;
    case TINYGLTF_TYPE_MAT3:
        return ElementType::Matrix3x3;
        break;
    case TINYGLTF_TYPE_MAT4:
        return ElementType::Matrix4x4;
        break;
    case TINYGLTF_TYPE_SCALAR:
        return ElementType::Scalar;
        break;
    case TINYGLTF_TYPE_VECTOR:
        return ElementType::Vector;
        break;
    case TINYGLTF_TYPE_MATRIX:
        return ElementType::Matrix;
        break;
    default:
        return ElementType::Scalar;
    }
}

PrimitiveMode ConvertGLTFPrimitiveMode(int32_t mode)
{
    switch (mode) {
    case  TINYGLTF_MODE_POINTS:
        return PrimitiveMode::Points;
        break;
    case  TINYGLTF_MODE_LINE:
        return PrimitiveMode::Line;
        break;
    case  TINYGLTF_MODE_LINE_LOOP:
        return PrimitiveMode::LineLoop;
        break;
    case  TINYGLTF_MODE_LINE_STRIP:
        return PrimitiveMode::LineStrip;
        break;
    case  TINYGLTF_MODE_TRIANGLES:
        return PrimitiveMode::Triangles;
        break;
    case  TINYGLTF_MODE_TRIANGLE_STRIP:
        return PrimitiveMode::TriangleStrip;
        break;
    case  TINYGLTF_MODE_TRIANGLE_FAN:
        return PrimitiveMode::TriangleFan;
        break;
    default:
        return PrimitiveMode::Triangles;
        break;
    }
}

AttributeType ConvertAttributeName(std::string name)
{
    AttributeType type = AttributeType::Invalid;
    if (name == "POSITION") {
        type = AttributeType::Position;
    }
    else if (name == "NORMAL") {
        type = AttributeType::Normal;
    }
    else if (name == "TANGENT") {
        type = AttributeType::Tangent;
    }
    else if (name == "TEXCOORD_0") {
        type = AttributeType::TexCoord1;
    }
    else if (name == "TEXCOORD_1") {
        type = AttributeType::TexCoord2;
    }
    return type;

}


/**
 * @brief サンプラー値に変換する
 * @param value0 入力値0
 * @param value1 入力値1
 * @param type サンプラーの種別
 * @return サンプラー値
*/
int32_t ConvertGLTFSamplerToSampler(int32_t value0, int32_t value1, GLTFSamplerValueType type)
{
    if (type == GLTFSamplerValueType_Filter)
    {
        if (value0 == TINYGLTF_TEXTURE_FILTER_NEAREST && value1 == TINYGLTF_TEXTURE_FILTER_NEAREST) {
            return static_cast<int32_t>(SamplerFilter::MIN_MAG_MIP_POINT);
        } else if (value0 == TINYGLTF_TEXTURE_FILTER_NEAREST && value1 == TINYGLTF_TEXTURE_FILTER_LINEAR) {
            return static_cast<int32_t>(SamplerFilter::MIN_LINEAR_MAG_MIP_POINT);
        } else if (value0 == TINYGLTF_TEXTURE_FILTER_LINEAR && value1 == TINYGLTF_TEXTURE_FILTER_NEAREST) {
            return static_cast<int32_t>(SamplerFilter::MIN_POINT_MAG_MIP_LINEAR);
        } else if (value0 == TINYGLTF_TEXTURE_FILTER_LINEAR && value1 == TINYGLTF_TEXTURE_FILTER_LINEAR) {
            return static_cast<int32_t>(SamplerFilter::MIN_MAG_MIP_LINEAR);
        }
        return static_cast<int32_t>(SamplerFilter::MIN_MAG_MIP_LINEAR);
    }
    else if (type == GLTFSamplerValueType_Wrap)
    {
        switch (value0) {
        case TINYGLTF_TEXTURE_WRAP_REPEAT:
            return static_cast<int32_t>(SamplerAccessMode::Wrap);
            break;
        case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
            return static_cast<int32_t>(SamplerAccessMode::Clamp);
            break;
        case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
            return static_cast<int32_t>(SamplerAccessMode::Mirror);
            break;
        default:
            break;
        }
    }
    return -1;
}

template<typename T>
void AdapterIntArray(std::vector<uint32_t>& vec, const uint32_t index, const uint8_t* dataAddress, const size_t byteStride)
{
    T data = *(reinterpret_cast<const T*>(dataAddress + index * byteStride));
    vec.push_back(static_cast<uint32_t>(data));
}

void CalculateBoudingBox(Buffer& position_buffer, Buffer& index_buffer, float* box_max, float* box_min)
{

    std::vector<uint32_t> indices;
    {
        auto value_type = index_buffer.value_type;
        auto element_type = index_buffer.element_type;
        size_t byte_stride = CalculateByteStride(value_type, element_type);
        for (uint32_t i = 0; i < index_buffer.count; i++)
        {
            if (value_type == ValueType::Uint16)
            {
                AdapterIntArray<uint16_t>(indices, i, index_buffer.data.data(), byte_stride);
            }
            else if (value_type == ValueType::Uint32)
            {
                AdapterIntArray<uint32_t>(indices, i, index_buffer.data.data(), byte_stride);
            }
        }
    }

    float tmp_box_max[3] = { -100000.0f, -100000.0f, -1000000.0f };
    float tmp_box_min[3] = { 1000000.0f, 1000000.0f, 1000000.0f };
    {
        auto value_type = position_buffer.value_type;
        auto element_type = position_buffer.element_type;
        size_t byte_stride = CalculateByteStride(value_type, element_type);
        for (uint32_t i = 0; i < indices.size(); i++)
        {
            float position[3];
            AdapterFloatArrayInterface(position, indices.at(i), position_buffer.data.data(), byte_stride, value_type, ConvertElementTypeBitSize(element_type));

            tmp_box_max[0] = std::max(tmp_box_max[0], position[0]);
            tmp_box_max[1] = std::max(tmp_box_max[1], position[1]);
            tmp_box_max[2] = std::max(tmp_box_max[2], position[2]);

            tmp_box_min[0] = std::min(tmp_box_min[0], position[0]);
            tmp_box_min[1] = std::min(tmp_box_min[1], position[1]);
            tmp_box_min[2] = std::min(tmp_box_min[2], position[2]);
        }
    }

    box_max[0] = tmp_box_max[0];
    box_max[1] = tmp_box_max[1];
    box_max[2] = tmp_box_max[2];

    box_min[0] = tmp_box_min[0];
    box_min[1] = tmp_box_min[1];
    box_min[2] = tmp_box_min[2];
}

int LoadGltfModel(tinygltf::Model& model, const std::string& filename)
{
    tinygltf::TinyGLTF loader = {};
    std::string err = {};
    std::string warn = {};
    // assume ascii glTF.
    int ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename.c_str());

    if (!warn.empty()) {
        printf_s("Warn: %s\n", warn.c_str());
        return -1;
    }

    if (!err.empty()) {
        printf_s("Error: %s\n", err.c_str());
        return -1;
    }

    if (!ret) {
        printf_s("Failed to parse GLTF: %s\n", filename.c_str());
        return -1;
    }
    return 1;
}

void PerseGLTFNode(const tinygltf::Model& gltf_model, const tinygltf::Node& gltf_node, Patch& patch, Model& model, const std::string& file_name, const std::string& file_path)
{

    uint32_t gltf_mesh_id = gltf_node.mesh;
    if (gltf_model.meshes.size() < gltf_mesh_id)
    {
        // 子要素があれば再帰
        for (auto& child_id : gltf_node.children)
        {
            model.children.push_back(child_id);
            PerseGLTFNode(gltf_model, gltf_model.nodes.at(child_id), patch, patch.models.at(child_id), file_name, file_path);
        }
    }

    // メッシュを取得し、プリミティブ毎に値をパース
    const tinygltf::Mesh& gltf_mesh = gltf_model.meshes.at(gltf_mesh_id);
    for (uint32_t primitive_id = 0; primitive_id < gltf_mesh.primitives.size(); primitive_id++) {
        const tinygltf::Primitive& gltf_primitive = gltf_mesh.primitives.at(primitive_id);

        patch.meshes.resize(patch.meshes.size() + 1);
        model.mesh_id.push_back((uint32_t)(patch.meshes.size() - 1));
        Mesh& mesh = patch.meshes.back();
        mesh.name = file_name + "_m" + gltf_mesh.name + "_p" + std::to_string(primitive_id);
        mesh.primitive_mode = ConvertGLTFPrimitiveMode(gltf_primitive.mode);

        // index_buffer情報を設定
        {
            uint32_t index_buffer_id = gltf_primitive.indices;
            Buffer& buffer = patch.buffers.at(index_buffer_id);
            buffer.buffer_type = BufferType::IndexBuffer;
            buffer.name = file_name + "_ib_" + std::to_string(index_buffer_id);
            mesh.index_buffer_id = index_buffer_id;
        }

        // vertex_buffer情報を設定
        {
            for (const auto& gltf_attribute : gltf_primitive.attributes)
            {
                uint32_t vertexb_buffer_id = gltf_attribute.second;
                AttributeType type = ConvertAttributeName(gltf_attribute.first);
                mesh.vertex_buffer_id[type] = vertexb_buffer_id;
                Buffer& buffer = patch.buffers.at(gltf_attribute.second);
                buffer.buffer_type = BufferType::VertexBuffer;
                buffer.name = file_name + "_vb" + std::to_string(vertexb_buffer_id);
            }
        }

        // material情報を格納
        {
            int32_t material_id = gltf_primitive.material;
            mesh.material_id = material_id;
        }

        // aabbを生成
        {
            Buffer& position_buffer = patch.buffers.at(mesh.vertex_buffer_id[AttributeType::Position]);
            Buffer& index_buffer = patch.buffers.at(mesh.index_buffer_id);
            CalculateBoudingBox(position_buffer, index_buffer, mesh.box_max, mesh.box_min);
        }
    }

    // Transform情報
    {
        std::string transform_name = file_name +"_trs_" + std::to_string(gltf_node.mesh);
        patch.transforms.resize(patch.transforms.size() + 1);
        uint32_t transform_id = (uint32_t)patch.transforms.size() - 1;
        Transform& transform = patch.transforms.at(transform_id);
        model.transform_id = transform_id;
        transform.name = transform_name;

        // 位置、回転、拡縮要素を格納
        if (gltf_node.matrix.empty())
        {
            if (!gltf_node.translation.empty())
            {
                transform.translate[0] = (float)gltf_node.translation.at(0);
                transform.translate[1] = (float)gltf_node.translation.at(1);
                transform.translate[2] = (float)gltf_node.translation.at(2);
            }

            if (!gltf_node.rotation.empty())
            {
                transform.rotation[0] = (float)gltf_node.rotation.at(0);
                transform.rotation[1] = (float)gltf_node.rotation.at(1);
                transform.rotation[2] = (float)gltf_node.rotation.at(2);
                transform.rotation[3] = (float)gltf_node.rotation.at(3);
            }

            if (!gltf_node.scale.empty())
            {
                transform.scale[0] = (float)gltf_node.scale.at(0);
                transform.scale[1] = (float)gltf_node.scale.at(1);
                transform.scale[2] = (float)gltf_node.scale.at(2);
            }
            transform.is_matrix = false;
        }
        else
        {
            // 行列情報を格納
            for (size_t i = 0; i < gltf_node.matrix.size(); i++)
            {
                transform.world[i] = (float)gltf_node.matrix.at(i);
            }
            transform.is_matrix = true;
        }
    }
    
    // 子要素があれば再帰
    for (auto& child_id : gltf_node.children) {
        model.children.push_back(child_id);
        PerseGLTFNode(gltf_model, gltf_model.nodes.at(child_id), patch, patch.models.at(child_id), file_name, file_path);
    }
}

void PerseGLTFMaterial(const tinygltf::Model& gltf_model, Patch& patch, const std::string& file_name)
{
    patch.materials.resize(gltf_model.materials.size());
    for (size_t i = 0; i < gltf_model.materials.size(); i++) {
        const tinygltf::Material& gltf_material = gltf_model.materials.at(i);
        Material& material = patch.materials.at(i);

        std::string material_name = file_name + "_material_" + std::to_string(i);

        material.name = material_name;

        // base_color
        {
            CustomParam4f custom = {};
            custom.name = "base_color";
            custom.parameter[0] = static_cast<float>(gltf_material.pbrMetallicRoughness.baseColorFactor.at(0));
            custom.parameter[1] = static_cast<float>(gltf_material.pbrMetallicRoughness.baseColorFactor.at(1));
            custom.parameter[2] = static_cast<float>(gltf_material.pbrMetallicRoughness.baseColorFactor.at(2));
            custom.parameter[3] = static_cast<float>(gltf_material.alphaCutoff);
            material.c_param4f.push_back(custom);
        }

        // emissive
        {
            CustomParam4f custom = {};
            custom.name = "emissive";
            custom.parameter[0] = static_cast<float>(gltf_material.emissiveFactor.at(0));
            custom.parameter[1] = static_cast<float>(gltf_material.emissiveFactor.at(1));
            custom.parameter[2] = static_cast<float>(gltf_material.emissiveFactor.at(2));
            custom.parameter[2] = 0.0f;
            material.c_param4f.push_back(custom);
        }

        // roughness
        {
            CustomParamf custom = {};
            custom.name = "rougness";
            custom.parameter = static_cast<float>(gltf_material.pbrMetallicRoughness.roughnessFactor);
            material.c_paramf.push_back(custom);
        }

        // metalness
        {
            CustomParamf custom = {};
            custom.name = "metalness";
            custom.parameter = static_cast<float>(gltf_material.pbrMetallicRoughness.metallicFactor);
            material.c_paramf.push_back(custom);
        }

        // texcoord
        {
            int32_t tex_id[5] = {
                gltf_material.pbrMetallicRoughness.baseColorTexture.texCoord,
                gltf_material.normalTexture.texCoord,
                gltf_material.pbrMetallicRoughness.metallicRoughnessTexture.texCoord,
                gltf_material.occlusionTexture.texCoord,
                gltf_material.emissiveTexture.texCoord
            };

            std::string custom_names[5] = {
                "base_color_tex_id",
                "normal_tex_id",
                "metal_roughness_tex_id",
                "occlusion_tex_id",
                "emissive_tex_id",
            };

            for (uint32_t j = 0; j < 5; j++) {
                CustomParami custom = {};
                custom.name = custom_names[j];
                custom.parameter = tex_id[j];
                material.c_parami.push_back(custom);
            }
        }

        // テクスチャ,サンプラー情報をパース
        size_t texture_count = 0;
        {
            int32_t texture_indices[5] = {};
            texture_indices[0] = gltf_material.pbrMetallicRoughness.baseColorTexture.index;
            texture_indices[1] = gltf_material.pbrMetallicRoughness.metallicRoughnessTexture.index;
            texture_indices[2] = gltf_material.normalTexture.index;
            texture_indices[3] = gltf_material.occlusionTexture.index;
            texture_indices[4] = gltf_material.emissiveTexture.index;

            for (uint32_t j = 0; j < 5; j++) {
                if (!(texture_indices[j] >= 0 && texture_indices[j] < gltf_model.textures.size())) {
                    continue;
                }
                texture_count++;
                material.texture_ids.push_back(texture_indices[j]);
            }
        }

        //shader
        {
            Shader shader = {};
            material.shader.name = "gltf_pbr_" + file_name + "_tex" + std::to_string(texture_count);
            material.shader.path = "model/";
        }
    }
}

void PerseGLTFTexture(const tinygltf::Model& gltf_model, Patch& patch, const std::string& file_path)
{
    patch.textures.resize(gltf_model.textures.size());
    for (size_t i = 0; i < gltf_model.textures.size(); i++) {
        const tinygltf::Texture& gltf_texture = gltf_model.textures.at(i);
        const tinygltf::Image& gltf_image = gltf_model.images.at(gltf_texture.source);

        Texture& texture = patch.textures.at(i);
        const std::string texture_name = gltf_image.uri;
        const std::string texture_path = file_path + texture_name;
        texture.name = texture_name;
        texture.path = texture_path;
        texture.sampler_id = std::max(gltf_texture.sampler, 0);
    }
}

void PerseGLTFSampler(const tinygltf::Model& gltf_model, Patch& patch)
{
    patch.samplers.resize(gltf_model.samplers.size());
    for (size_t i = 0; i < gltf_model.samplers.size(); i++) {
        const tinygltf::Sampler& gltf_sampler = gltf_model.samplers.at(i);
        Sampler sampler = patch.samplers.at(i);
        sampler.filter = ConvertGLTFSamplerToSampler(gltf_sampler.magFilter, gltf_sampler.minFilter, GLTFSamplerValueType_Filter);
        sampler.address_u = ConvertGLTFSamplerToSampler(gltf_sampler.wrapS, 0, GLTFSamplerValueType_Wrap);
        sampler.address_v = ConvertGLTFSamplerToSampler(gltf_sampler.wrapT, 0, GLTFSamplerValueType_Wrap);
        sampler.address_w = ConvertGLTFSamplerToSampler(gltf_sampler.wrapT, 0, GLTFSamplerValueType_Wrap);
    }
}

void PerseGLTFBuffer(const tinygltf::Model& gltf_model, Patch& patch, const std::string& file_name)
{
    patch.buffers.resize(gltf_model.accessors.size());
    for (size_t i = 0; i < gltf_model.accessors.size(); i++) {
        const tinygltf::Accessor& gltf_accessor = gltf_model.accessors.at(i);
        const tinygltf::BufferView& gltf_buffer_view = gltf_model.bufferViews[gltf_accessor.bufferView];
        const tinygltf::Buffer& gltf_buffer = gltf_model.buffers[gltf_buffer_view.buffer];
        const uint8_t* data_address = gltf_buffer.data.data() + gltf_buffer_view.byteOffset + gltf_accessor.byteOffset;
        const uint32_t byte_stride = gltf_accessor.ByteStride(gltf_buffer_view);
        const uint32_t count = (uint32_t)gltf_accessor.count;
        const uint32_t byte_size = count * byte_stride;

        Buffer& buffer = patch.buffers.at(i);
        buffer.name = file_name + "_b_" + std::to_string(i);
        buffer.buffer_type = BufferType::VertexBuffer;
        buffer.element_type = ConvertGLTFElementType(gltf_accessor.type);
        buffer.value_type = ConvertGLTFValueType(gltf_accessor.componentType);
        buffer.count = (uint32_t)gltf_accessor.count;
        buffer.data.resize(byte_size);
        memcpy(buffer.data.data(), data_address, byte_size);
    }
}

GltfPerser::GltfPerser()
{

}

GltfPerser::~GltfPerser()
{

}

bool GltfPerser::Perse(Patch& patch, const std::string& input_path)
{
    tinygltf::Model gltf_model = {};
    int ret = LoadGltfModel(gltf_model, input_path);

    if (ret < 0) {
        return false;
    }

    std::string file_path = input_path;
    std::string file_name = input_path;
    size_t last_back_slush = input_path.find_last_of("\\");
    file_path.replace(last_back_slush, std::string::npos, "\\");
    file_name.replace(0, last_back_slush + 1, "");
    size_t last_dot = file_name.find_last_of(".");
    file_name.replace(last_dot, std::string::npos, "");
    PerseGLTFMaterial(gltf_model, patch, file_name);
    PerseGLTFTexture(gltf_model, patch, file_path);
    PerseGLTFSampler(gltf_model, patch);
    PerseGLTFBuffer(gltf_model, patch, file_name);

    patch.name = file_name;
    patch.models.resize(gltf_model.nodes.size());
    if (gltf_model.nodes.size() > 0) {
        patch.models.at(0).name = file_name;
        PerseGLTFNode(gltf_model, gltf_model.nodes.at(0), patch, patch.models.at(0), file_name, file_path);
    }
    return true;
}
