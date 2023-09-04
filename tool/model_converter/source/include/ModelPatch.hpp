#pragma once

#include <string>
#include <array>
#include <vector>
#include <memory>
#include <map>

#define CATCH_CONFIG_MAIN

#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/map.hpp>

enum class SamplerFilter : uint32_t{
    MIN_MAG_MIP_POINT = 0,
    MIN_MAG_POINT_MIP_LINEAR = 0x1,
    MIN_POINT_MAG_LINEAR_MIP_POINT = 0x4,
    MIN_POINT_MAG_MIP_LINEAR = 0x5,
    MIN_LINEAR_MAG_MIP_POINT = 0x10,
    MIN_LINEAR_MAG_POINT_MIP_LINEAR = 0x11,
    MIN_MAG_LINEAR_MIP_POINT = 0x14,
    MIN_MAG_MIP_LINEAR = 0x15,
    ANISOTROPIC = 0x55,
    COMPARISON_MIN_MAG_MIP_POINT = 0x80,
    COMPARISON_MIN_MAG_POINT_MIP_LINEAR = 0x81,
    COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT = 0x84,
    COMPARISON_MIN_POINT_MAG_MIP_LINEAR = 0x85,
    COMPARISON_MIN_LINEAR_MAG_MIP_POINT = 0x90,
    COMPARISON_MIN_LINEAR_MAG_POINT_MIP_LINEAR = 0x91,
    COMPARISON_MIN_MAG_LINEAR_MIP_POINT = 0x94,
    COMPARISON_MIN_MAG_MIP_LINEAR = 0x95,
    COMPARISON_ANISOTROPIC = 0xd5,
    MINIMUM_MIN_MAG_MIP_POINT = 0x100,
    MINIMUM_MIN_MAG_POINT_MIP_LINEAR = 0x101,
    MINIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT = 0x104,
    MINIMUM_MIN_POINT_MAG_MIP_LINEAR = 0x105,
    MINIMUM_MIN_LINEAR_MAG_MIP_POINT = 0x110,
    MINIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR = 0x111,
    MINIMUM_MIN_MAG_LINEAR_MIP_POINT = 0x114,
    MINIMUM_MIN_MAG_MIP_LINEAR = 0x115,
    MINIMUM_ANISOTROPIC = 0x155,
    MAXIMUM_MIN_MAG_MIP_POINT = 0x180,
    MAXIMUM_MIN_MAG_POINT_MIP_LINEAR = 0x181,
    MAXIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT = 0x184,
    MAXIMUM_MIN_POINT_MAG_MIP_LINEAR = 0x185,
    MAXIMUM_MIN_LINEAR_MAG_MIP_POINT = 0x190,
    MAXIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR = 0x191,
    MAXIMUM_MIN_MAG_LINEAR_MIP_POINT = 0x194,
    MAXIMUM_MIN_MAG_MIP_LINEAR = 0x195,
    MAXIMUM_ANISOTROPIC = 0x1d5
};

enum class SamplerAccessMode : uint32_t {
    Wrap = 1,
    Mirror = 2,
    Clamp = 3,
    Border = 4,
    MirrorOnce = 5
};

enum class ValueType : uint32_t {
    Invalid,
    Uint8,
    Sint8,
    Uint16,
    Sint16,
    Uint32,
    Sint32,
    Uint128,
    Sint128,
    Float,
    Double,
    LDouble
};

enum class ElementType : uint32_t {
    Invalid,
    Vector1,
    Vector2,
    Vector3,
    Vector4,
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
    Scalar,
    Vector,
    Matrix
};


enum class BufferType : uint32_t {
    VertexBuffer,
    IndexBuffer,
};

enum class PrimitiveMode : uint32_t {
    Points,
    Line,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan
};

enum class AttributeType : uint32_t {
    Invalid = 0,
    Position,
    PrevPosition,
    TexCoord1,
    TexCoord2,
    Normal,
    Material,
    Tangent,
    Transform,
    PrevTransform,
    JointIndices,
    JointWeights,
    Count
};

struct Shader {
    std::string path = "";
    std::string name = "";

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("path", path),
            cereal::make_nvp("name", name)
        );
    }
};

struct Sampler {
    int32_t filter = 0;
    int32_t address_u = 0;
    int32_t address_v = 0;
    int32_t address_w = 0;

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("filter", filter),
            cereal::make_nvp("adddress_u", address_u),
            cereal::make_nvp("adddress_v", address_v),
            cereal::make_nvp("adddress_w", address_w)
        );
    }

};

struct Texture {
    int32_t sampler_id = 0;
    std::string path = "";
    std::string name = "";

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("sampler_id", sampler_id),
            cereal::make_nvp("path", path),
            cereal::make_nvp("name", name)
        );
    }
};


struct CustomParam4f {
    float parameter[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    std::string name = "";

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("parameter", parameter),
            cereal::make_nvp("name", name)
        );
    }
};

struct CustomParam4i {
    int32_t parameter[4] = {0, 0, 0, 0};
    std::string name = "";

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("parameter", parameter),
            cereal::make_nvp("name", name)
        );
    }
};

struct CustomParamf {
    float parameter = 0.0f;
    std::string name = "";

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("parameter", parameter),
            cereal::make_nvp("name", name)
        );
    }
};

struct CustomParami {
    int parameter = 0;
    std::string name = "";

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("parameter", parameter),
            cereal::make_nvp("name", name)
        );
    }
};

struct Material {
    std::string name = "";
    Shader shader = {};
    std::vector<CustomParam4f>  c_param4f = {};
    std::vector<CustomParam4i>  c_param4i = {};
    std::vector<CustomParamf>   c_paramf = {};
    std::vector<CustomParami>   c_parami = {};
    std::vector<uint32_t> texture_ids;
    uint32_t flag = 0;
    uint32_t stencil_mask = 0;

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("name", name),
            cereal::make_nvp("shader", shader),
            cereal::make_nvp("param_float4", c_param4f),
            cereal::make_nvp("param_int4", c_param4i),
            cereal::make_nvp("param_float", c_paramf),
            cereal::make_nvp("param_int", c_parami),
            cereal::make_nvp("texture_ids", texture_ids),
            cereal::make_nvp("stencil", stencil_mask),
            cereal::make_nvp("flag", flag)
        );
    }
};

struct Transform {
    std::string name = "";
    bool is_matrix = false;
    float translate[3] = {0.0f, 0.0f, 0.0f};
    float rotation[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float scale[3] = {1.0f, 1.0f, 1.0f};
    float world[16] = {};
    float max_pos[3] = {0.0f, 0.0f, 0.0f};
    float min_pos[3] = {0.0f, 0.0f, 0.0f};

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("name", name),
            cereal::make_nvp("is_matrix", is_matrix),
            cereal::make_nvp("translate", translate),
            cereal::make_nvp("rotation", rotation),
            cereal::make_nvp("scale", scale),
            cereal::make_nvp("world", world),
            cereal::make_nvp("max_pos", max_pos),
            cereal::make_nvp("min_pos", min_pos)
        );
    }
};

struct Buffer {
    BufferType buffer_type = {};
    ValueType value_type = {};
    ElementType element_type = {};
    uint32_t count = 0;
    std::string name = {};
    std::vector<uint8_t> data = {};

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("buffer_type", buffer_type),
            cereal::make_nvp("value_type", value_type),
            cereal::make_nvp("element_type", element_type),
            cereal::make_nvp("count", count),
            cereal::make_nvp("name", name),
            cereal::make_nvp("data", data)
        );
    }
};


struct Mesh
{
    std::string name = "";
    uint32_t index_buffer_id = {};
    std::map<AttributeType, uint32_t> vertex_buffer_id = {};
    uint32_t material_id = {};
    PrimitiveMode primitive_mode = {};
    float box_max[3] = { 1000.0f, 1000.0f, 1000.0f };
    float box_min[3] = { -1000.0f, -1000.0f, -1000.0f };
    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("name", name),
            cereal::make_nvp("index_buffer_id", index_buffer_id),
            cereal::make_nvp("vertex_buffer_id", vertex_buffer_id),
            cereal::make_nvp("material_id", material_id),
            cereal::make_nvp("primitive_mode", primitive_mode),
            cereal::make_nvp("box_max", box_max),
            cereal::make_nvp("box_min", box_min)
        );
    }

};

struct Model {
    std::vector<uint32_t> children = {};
    std::vector<uint32_t> mesh_id = {};
    uint32_t transform_id = {};
    std::string name = {};
    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("children", children),
            cereal::make_nvp("mesh_id", mesh_id),
            cereal::make_nvp("transform_id", transform_id),
            cereal::make_nvp("name", name)
        );
    }
};

struct Patch {
    std::string                 name = {};
    std::vector<Model>          models = {};
    std::vector<Texture>        textures = {};
    std::vector<Sampler>        samplers = {};
    std::vector<Material>       materials = {};
    std::vector<Transform>      transforms = {};
    std::vector<Buffer>         buffers = {};
    std::vector<Mesh>           meshes;

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(
            cereal::make_nvp("name", name),
            cereal::make_nvp("models", models),
            cereal::make_nvp("textures", textures),
            cereal::make_nvp("samplers", samplers),
            cereal::make_nvp("materials", materials),
            cereal::make_nvp("transform", transforms),
            cereal::make_nvp("buffers", buffers),
            cereal::make_nvp("meshes", meshes)
        );
    }
};
