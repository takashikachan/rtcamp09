/**
 * @file    GpuResource.hpp
 * @brief   GPUリソースｎ定義
 */

#pragma once


#include "graphic/GraphicsContext.hpp"
#include "graphic/ObjectData.hpp"

namespace slug
{
/**
 * @brief 頂点属性
*/
enum class VertexAttribute
{
    Position,
    Normal,
    Texcoord,
    Index,
    MaterialIndex,
};

/**
 * @brief CUDA用のバッファ
*/
struct CudaBuffer
{
    std::string name = {};                              //!< 名前
    size_t byte_size = 0;                               //!< バイトサイズ
    CUdeviceptr buffer = {};                            //!< バッファポインタ
    VertexAttribute attribute = {};                     //!< 頂点要素の種類
    data::ValueType value_type = data::ValueType::Float;            //!< 値の種類
    data::ElementType element_type = data::ElementType::Vector1;    //!< 要素の種類

    void CreateBuffer(GraphicsContext& context, size_t _byte_size, data::ValueType _type, data::ElementType _element, std::string _name, void* data) 
    {
        name = _name;
        byte_size = _byte_size;
        value_type = _type;
        element_type = _element;
        context.CreateBuffer(buffer, byte_size, data);
    }
};

/**
 * @brief CUDA用のテクスチャ
*/
struct CudaTexture
{
    std::string name = {};                              //!< 名前
    cudaTextureObject_t texObj = {};                    //!< テクスチャオブジェクト
    cudaArray_t texArray;                               //!< テクスチャの実データ
};

/**
 * @brief ジオメトリ加速構造体
*/
struct GeometryAS 
{
    std::string name = {};                              //!< 名前
    OptixTraversableHandle handle = {};                 //!< ハンドル
    CUdeviceptr output_buffer = {};                     //!< 実データ
    uint32_t vertex_buffer_id = {};                     //!< 頂点バッファID
    uint32_t index_buffer_id = {};                      //!< indexバッファID
    uint32_t offset_buffer_id = {};                     //!< オフセットバッファID
    uint32_t normal_buffer_id = {};                     //!< 法線
    uint32_t texcoord_buffer_id = {};                   //!< uv
    uint32_t material_buffer_id = {};                   //!< マテリアルバッファID
    uint32_t bind_instance_as_id = {};
};

/**
 * @brief インスタンス加速構造体
*/
struct InstanceAS 
{
    std::string name = {};                              //!< 名前
    OptixInstance instance = {};                        //!< インスタンス情報
    Transform transform = {};
    uint32_t bind_geometry_as_id = {};                  //!< 紐づいているジオメトリ加速構造体のID
};

/**
 * @brief インスタンスグループ
*/
struct InstanceASGroup 
{
    std::string name = {};                          //!< 名前
    OptixTraversableHandle handle = {};             //!< ハンドル
    CUdeviceptr output_buffer = {};                 //!< 実データ
};

struct CudaMaterial
{
    float base_color[3] = {1.0f, 1.0f, 1.0f};
    float emission[3] = { 0.0f, 0.0f, 0.0f };
    float ior = 1.0f;
    float relative_ior = 1.0f;
    float specular_tint = 0.0f;
    float specular_trans = 0.0f;
    float sheen = 0.0f;
    float sheen_tint = 0.0f;
    float roughness = 0.0f;
    float metallic = 0.0f;
    float clearcoat = 0.0f;
    float clearcoat_gloss = 0.0f;
    float subsurface = 0.0f;
    float anisotropic = 0.0f;
    float debug_diffuse = 1.0f;
    float debug_specular = 1.0f;
    cudaTextureObject_t albedo;
    cudaTextureObject_t bump;

};
enum LightType
{
    Sphere,
};


struct CudaLight 
{
    LightType type;
    float3 position;
    float radius;
    float3 emission;
};


struct ResoucePool
{
    std::vector<CudaBuffer> buffer_table = {};
    std::vector<CudaTexture> texture_table = {};
    std::vector<CudaMaterial> material_table = {};
    std::vector<CudaLight> light_table = {};
    std::vector<GeometryAS> geometry_as_table = {};
    std::vector<InstanceAS> instance_as_table = {};
    InstanceASGroup root_instance_as = {};
    CudaTexture envrionment_texture = {};

    template<typename T>
    bool Add(T& vector, std::string& name, uint32_t& index)
    {
        bool found = false;
        for (size_t i = 0; i < vector.size(); i++)
        {
            if (vector.at(i).name == name)
            {
                index = static_cast<uint32_t>(i);
                found = true;
                break;
            }
        }

        if (!found)
        {
            vector.resize(vector.size() + 1);
            index = (uint32_t)(vector.size() - 1);
        }
        return found;
    }
};

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

extern size_t ConvertValueTypeBitSize(data::ValueType value_type);

extern size_t ConvertElementTypeBitSize(data::ElementType element_type);

extern OptixVertexFormat ConvertOptixVertexFormat(data::ValueType value_type, data::ElementType element_type);

extern OptixIndicesFormat ConvertIndincesFormat(data::ValueType value_type);

extern size_t CalcBufferBitSize(data::ValueType value_type, data::ElementType element_type, uint32_t count);

extern size_t CalcBufferStride(data::ValueType value_type, data::ElementType element_type);

extern void UpdateTransform(InstanceAS& instanceAS);

extern bool ConvertAttribute(data::AttributeType type, VertexAttribute& value);

extern void CreateGeometryResource(GraphicsContext& context, data::Scene& object, data::Mesh& mesh, data::Model& model, std::vector<uint32_t>& instance_as_ids, ResoucePool& resouce_pool);

extern bool CreateModelResouce(GraphicsContext& context, data::Scene& object, data::Model& model, std::string parent_as_name, ResoucePool& resouce_pool);

extern void CreateMaterialResource(data::Scene& object, ResoucePool& resouce_pool);

extern void CreateTextureResouce(data::Scene& object, GraphicsContext& context, ResoucePool& resouce_pool);

extern void CreateEnvironmentResouce(GraphicsContext& context, ResoucePool& resouce_pool);

extern void CreateObject(data::Scene& object, GraphicsContext& context, ResoucePool& resouce_pool);

extern void CreatRootInstance(GraphicsContext& context, ResoucePool& resouce_pool, CUdeviceptr& d_temp_buffer, CUdeviceptr& d_comapct_tmp_buffer);

extern void UpdateRootInstance(GraphicsContext& context, ResoucePool& resouce_pool, CUstream& cu_stream, CUdeviceptr& d_temp_buffer, CUdeviceptr& d_comapct_tmp_buffer);
} // namespace slug