/**
 * @file    ObjPerser.cpp
 * @brief   Objのパースを行うクラスのソース
 *
 */

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "ObjPerser.hpp"
#include "Utility.hpp"
#include <algorithm>
#include <unordered_map>

struct vec3 {
    float v[3] = { 0.0f, 0.0f, 0.0f };
};

std::map<std::string, uint32_t> texture_map = {};


ObjPerser::ObjPerser()
{

}
ObjPerser::~ObjPerser()
{

}

void ComputeSmoothingShape(const tinyobj::attrib_t& inattrib,const tinyobj::shape_t& inshape,
                                  std::vector<std::pair<unsigned int, unsigned int>>& sortedids,
                                  unsigned int idbegin, unsigned int idend,
                                  std::vector<tinyobj::shape_t>& outshapes,
                                  tinyobj::attrib_t& outattrib)
{
    unsigned int sgroupid = sortedids[idbegin].first;
    bool hasmaterials = inshape.mesh.material_ids.size();
    outshapes.emplace_back();
    tinyobj::shape_t& outshape = outshapes.back();
    outshape.name = inshape.name;

    std::unordered_map<unsigned int, unsigned int> remap;
    for (unsigned int id = idbegin; id < idend; ++id)
    {
        unsigned int face = sortedids[id].second;

        outshape.mesh.num_face_vertices.push_back(3); // always triangles
        if (hasmaterials) {
            outshape.mesh.material_ids.push_back(inshape.mesh.material_ids[face]);
        }
        outshape.mesh.smoothing_group_ids.push_back(sgroupid);

        for (unsigned int v = 0; v < 3; ++v)
        {
            tinyobj::index_t inidx = inshape.mesh.indices[3 * face + v], outidx;
            assert(inidx.vertex_index != -1);
            auto iter = remap.find(inidx.vertex_index);

            if (sgroupid && iter != remap.end())
            {
                outidx.vertex_index = (*iter).second;
                outidx.normal_index = outidx.vertex_index;
                outidx.texcoord_index = (inidx.texcoord_index == -1) ? -1 : outidx.vertex_index;
            }
            else
            {
                assert(outattrib.vertices.size() % 3 == 0);
                unsigned int offset = static_cast<unsigned int>(outattrib.vertices.size() / 3);
                outidx.vertex_index = outidx.normal_index = offset;
                outidx.texcoord_index = (inidx.texcoord_index == -1) ? -1 : offset;
                outattrib.vertices.push_back(inattrib.vertices[3 * inidx.vertex_index]);
                outattrib.vertices.push_back(inattrib.vertices[3 * inidx.vertex_index + 1]);
                outattrib.vertices.push_back(inattrib.vertices[3 * inidx.vertex_index + 2]);
                outattrib.normals.push_back(0.0f);
                outattrib.normals.push_back(0.0f);
                outattrib.normals.push_back(0.0f);
                if (inidx.texcoord_index != -1)
                {
                    outattrib.texcoords.push_back(inattrib.texcoords[2 * inidx.texcoord_index]);
                    outattrib.texcoords.push_back(inattrib.texcoords[2 * inidx.texcoord_index + 1]);
                }
                remap[inidx.vertex_index] = offset;
            }
            outshape.mesh.indices.push_back(outidx);
        }
    }
}

void ComputeSmoothingShapes(const tinyobj::attrib_t& inattrib,
                                   const std::vector<tinyobj::shape_t>& inshapes,
                                   std::vector<tinyobj::shape_t>& outshapes,
                                   tinyobj::attrib_t& outattrib)
{
    for (size_t s = 0, slen = inshapes.size(); s < slen; ++s)
    {
        const tinyobj::shape_t& inshape = inshapes[s];

        unsigned int numfaces = static_cast<unsigned int>(inshape.mesh.smoothing_group_ids.size());
        assert(numfaces);
        std::vector<std::pair<unsigned int, unsigned int>> sortedids(numfaces);
        for (unsigned int i = 0; i < numfaces; ++i) {
            sortedids[i] = std::make_pair(inshape.mesh.smoothing_group_ids[i], i);
        }
        std::sort(sortedids.begin(), sortedids.end());

        unsigned int activeid = sortedids[0].first;
        unsigned int id = activeid, idbegin = 0, idend = 0;
        while (idbegin < numfaces)
        {
            while (activeid == id && ++idend < numfaces) {
                id = sortedids[idend].first;
            }
            ComputeSmoothingShape(inattrib, inshape, sortedids, idbegin, idend, outshapes, outattrib);
            activeid = id;
            idbegin = idend;
        }
    }
}

void ComputeAllSmoothingNormals(tinyobj::attrib_t& attrib,
                                       std::vector<tinyobj::shape_t>& shapes)
{
    vec3 p[3];
    for (size_t s = 0, slen = shapes.size(); s < slen; ++s)
    {
        const tinyobj::shape_t& shape(shapes[s]);
        size_t facecount = shape.mesh.num_face_vertices.size();
        assert(shape.mesh.smoothing_group_ids.size());

        for (size_t f = 0, flen = facecount; f < flen; ++f)
        {
            for (unsigned int v = 0; v < 3; ++v) {
                tinyobj::index_t idx = shape.mesh.indices[3 * f + v];
                assert(idx.vertex_index != -1);
                p[v].v[0] = attrib.vertices[3 * idx.vertex_index];
                p[v].v[1] = attrib.vertices[3 * idx.vertex_index + 1];
                p[v].v[2] = attrib.vertices[3 * idx.vertex_index + 2];
            }

            // cross(p[1] - p[0], p[2] - p[0])
            float nx = (p[1].v[1] - p[0].v[1]) * (p[2].v[2] - p[0].v[2]) -
                (p[1].v[2] - p[0].v[2]) * (p[2].v[1] - p[0].v[1]);
            float ny = (p[1].v[2] - p[0].v[2]) * (p[2].v[0] - p[0].v[0]) -
                (p[1].v[0] - p[0].v[0]) * (p[2].v[2] - p[0].v[2]);
            float nz = (p[1].v[0] - p[0].v[0]) * (p[2].v[1] - p[0].v[1]) -
                (p[1].v[1] - p[0].v[1]) * (p[2].v[0] - p[0].v[0]);

            // Don't normalize here.
            for (unsigned int v = 0; v < 3; ++v) {
                tinyobj::index_t idx = shape.mesh.indices[3 * f + v];
                attrib.normals[3 * idx.normal_index] += nx;
                attrib.normals[3 * idx.normal_index + 1] += ny;
                attrib.normals[3 * idx.normal_index + 2] += nz;
            }
        }
    }

    assert(attrib.normals.size() % 3 == 0);
    for (size_t i = 0, nlen = attrib.normals.size() / 3; i < nlen; ++i) {
        tinyobj::real_t& nx = attrib.normals[3 * i];
        tinyobj::real_t& ny = attrib.normals[3 * i + 1];
        tinyobj::real_t& nz = attrib.normals[3 * i + 2];
        tinyobj::real_t len = sqrtf(nx * nx + ny * ny + nz * nz);
        tinyobj::real_t scale = len == 0 ? 0 : 1 / len;
        nx *= scale;
        ny *= scale;
        nz *= scale;
    }
}

bool LoadObj(tinyobj::attrib_t& attrib, std::vector < tinyobj::shape_t>& shapes, std::vector<tinyobj::material_t>& materials, std::string filename, std::string filepath)
{
    
    tinyobj::ObjReaderConfig reader_config;
    reader_config.triangulate = true;
    reader_config.triangulation_method = "earcut";
    reader_config.vertex_color = false;
    reader_config.mtl_search_path = filepath;

    //bool ret = tinyobj::LoadObj(&inattrib, &inshapes, &materials, &warn, &err, filename.c_str(), filepath.c_str(), true);
    tinyobj::ObjReader reader;
    bool ret = reader.ParseFromFile(filename, reader_config);
    auto& inattrib = reader.GetAttrib();
    auto& inshapes = reader.GetShapes();
    materials = reader.GetMaterials();

    ComputeSmoothingShapes(inattrib, inshapes, shapes, attrib);
    ComputeAllSmoothingNormals(attrib, shapes);

    return ret;
}

void PerseAttribute(Patch& patch, tinyobj::attrib_t& attrib, std::string filename)
{

    if (attrib.vertices.size() > 0) {
        patch.buffers.resize(patch.buffers.size() + 1);
        Buffer& buffer = patch.buffers.back();
        buffer.name = filename + "vb0";
        buffer.buffer_type = BufferType::VertexBuffer;
        buffer.value_type = ValueType::Float;
        buffer.element_type = ElementType::Vector3;
        buffer.count = (uint32_t)attrib.vertices.size() / 3;
        uint32_t byte_size = (uint32_t)attrib.vertices.size() * sizeof(float);
        buffer.data.resize(byte_size);
        memcpy(buffer.data.data(), attrib.vertices.data(), byte_size);
    }


    if (attrib.texcoords.size() > 0) {
        patch.buffers.resize(patch.buffers.size() + 1);
        Buffer& buffer = patch.buffers.back();
        buffer.name = filename + "vb1";
        buffer.buffer_type = BufferType::VertexBuffer;
        buffer.value_type = ValueType::Float;
        buffer.element_type = ElementType::Vector2;
        buffer.count = (uint32_t)attrib.texcoords.size() / 2;
        uint32_t byte_size = (uint32_t)attrib.texcoords.size() * sizeof(float);
        buffer.data.resize(byte_size);
        memcpy(buffer.data.data(), attrib.texcoords.data(), byte_size);
    }


    if (attrib.normals.size() > 0) {
        patch.buffers.resize(patch.buffers.size() + 1);
        Buffer& buffer = patch.buffers.back();
        buffer.name = filename + "vb2";
        buffer.buffer_type = BufferType::VertexBuffer;
        buffer.value_type = ValueType::Float;
        buffer.element_type = ElementType::Vector3;
        buffer.count = (uint32_t)attrib.normals.size() / 3;
        uint32_t byte_size = (uint32_t)attrib.normals.size() * sizeof(float);
        buffer.data.resize(byte_size);
        memcpy(buffer.data.data(), attrib.normals.data(), byte_size);
    }

}

void PerseMaterial(Patch& patch, std::vector<tinyobj::material_t>& materials, std::string filename)
{
    auto SetCusutomParam4f = [](Material& material, std::string name, float* param, uint32_t size) {
        CustomParam4f custom = {};
        custom.name = name;
        for (uint32_t j = 0; j < size; j++) {
            custom.parameter[j] = param[j];
        }
        material.c_param4f.push_back(custom);
    };

    auto SetCusutomParamf = [](Material& material, std::string name, float param) {
        CustomParamf custom = {};
        custom.name = name;
        custom.parameter = param;
        material.c_paramf.push_back(custom);
    };

    auto SetTextureId = [](Material& material, std::string name, int32_t& texture_count) {
        if (!name.empty()) {
            if (texture_map.find(name) != texture_map.end()) {
                material.texture_ids.push_back(texture_map[name]);
                texture_count++;
            } else {
                uint32_t id = (uint32_t)texture_map.size();
                texture_map[name] = id;
                material.texture_ids.push_back(id);
                texture_count++;
            }
        }
    };

    patch.materials.resize(materials.size());
    for (uint32_t i = 0; i < materials.size(); i++) {
        int32_t texture_count = 0;

        tinyobj::material_t& obj_material = materials.at(i);
        Material& material = patch.materials.at(i);
        material.name = obj_material.name;

        SetCusutomParam4f(material, "ambient", obj_material.ambient, 3);
        SetCusutomParam4f(material, "diffuse", obj_material.diffuse, 3);
        SetCusutomParam4f(material, "specular", obj_material.specular, 3);
        SetCusutomParam4f(material, "transmittance", obj_material.transmittance, 3);
        SetCusutomParam4f(material, "emission", obj_material.emission, 3);
        SetCusutomParamf(material, "shininess", obj_material.shininess);
        SetCusutomParamf(material, "ior", obj_material.ior);
        SetCusutomParamf(material, "roughness", obj_material.roughness);
        SetCusutomParamf(material, "metalness", obj_material.metallic);
        SetCusutomParamf(material, "sheen", obj_material.sheen);
        SetCusutomParamf(material, "clearcoat_thickness", obj_material.clearcoat_roughness);
        SetCusutomParamf(material, "clearcoat_roughness", obj_material.clearcoat_roughness);
        SetCusutomParamf(material, "anisotropy", obj_material.anisotropy);
        SetCusutomParamf(material, "anisotropy_rotation", obj_material.anisotropy_rotation);

        SetTextureId(material, obj_material.ambient_texname, texture_count);
        SetTextureId(material, obj_material.diffuse_texname, texture_count);
        SetTextureId(material, obj_material.specular_texname, texture_count);
        SetTextureId(material, obj_material.specular_highlight_texname, texture_count);
        SetTextureId(material, obj_material.bump_texname, texture_count);
        SetTextureId(material, obj_material.displacement_texname, texture_count);
        SetTextureId(material, obj_material.alpha_texname, texture_count);
        SetTextureId(material, obj_material.reflection_texname, texture_count);
        SetTextureId(material, obj_material.roughness_texname, texture_count);
        SetTextureId(material, obj_material.metallic_texname, texture_count);
        SetTextureId(material, obj_material.emissive_texname, texture_count);
        SetTextureId(material, obj_material.reflection_texname, texture_count);
        SetTextureId(material, obj_material.normal_texname, texture_count);


        Shader shader = {};
        material.shader.name = "obj_pbr_" + filename + "_tex" + std::to_string(texture_count);
        material.shader.path = "model/";
    }
}

void PerseTexture(Patch& patch, std::string filepath)
{
    patch.textures.resize(texture_map.size());
    for (auto itr : texture_map) {
        Texture& texture = patch.textures.at(itr.second);
        texture.name = itr.first;
        texture.path = filepath + itr.first;
        texture.sampler_id = 0;
    }
}

void PerseShape(Patch& patch, Model& model, std::vector < tinyobj::shape_t>& shapes, std::string filename)
{

    for (uint32_t i = 0; i < shapes.size(); i++) {

        tinyobj::shape_t& shape = shapes.at(i);
        if (shape.mesh.indices.size() > 0)
        {
            Mesh mesh = {};


            int32_t index_buffer_id = 0;
            std::vector<int32_t> indices = {};
            indices.resize(shape.mesh.indices.size());
            for (uint32_t j = 0; j < shape.mesh.indices.size(); j++) {
                indices.at(j) = shape.mesh.indices.at(j).vertex_index;
            }
            {
                patch.buffers.resize(patch.buffers.size() + 1);
                Buffer& index_buffer = patch.buffers.back();
                index_buffer_id = (int32_t)patch.buffers.size() - 1;
                index_buffer.name = filename + "ib" + std::to_string(i);
                index_buffer.buffer_type = BufferType::IndexBuffer;
                index_buffer.value_type = ValueType::Uint32;
                index_buffer.element_type = ElementType::Scalar;
                index_buffer.count = (uint32_t)indices.size();
                uint32_t byte_size = (uint32_t)indices.size() * sizeof(int32_t);
                index_buffer.data.resize(byte_size);
                memcpy(index_buffer.data.data(), indices.data(), byte_size);
            }

            int32_t material_buffer_id = 0;
            {
                std::vector<int32_t> materials = {};
                materials.resize(shape.mesh.material_ids.size());
                for (uint32_t j = 0; j < shape.mesh.material_ids.size(); j++) {
                    materials.at(j) = shape.mesh.material_ids.at(j);
                }

                patch.buffers.resize(patch.buffers.size() + 1);
                Buffer& material_buffer = patch.buffers.back();
                material_buffer_id = (int32_t)patch.buffers.size() - 1;
                material_buffer.name = filename + "mb" + std::to_string(i);
                material_buffer.buffer_type = BufferType::VertexBuffer;
                material_buffer.value_type = ValueType::Uint32;
                material_buffer.element_type = ElementType::Scalar;
                material_buffer.count = (uint32_t)materials.size();
                uint32_t byte_size = (uint32_t)materials.size() * sizeof(int32_t);
                material_buffer.data.resize(byte_size);
                memcpy(material_buffer.data.data(), materials.data(), byte_size);
            }
            mesh.index_buffer_id = (uint32_t)index_buffer_id;
            mesh.material_id = material_buffer_id;
            mesh.name = shape.name;
            mesh.primitive_mode = PrimitiveMode::Triangles;
            mesh.vertex_buffer_id[AttributeType::Position] = 0;
            mesh.vertex_buffer_id[AttributeType::TexCoord1] = 1;
            mesh.vertex_buffer_id[AttributeType::Normal] = 2;
            mesh.vertex_buffer_id[AttributeType::Material] = material_buffer_id;
            model.mesh_id.push_back(i);
            patch.meshes.push_back(mesh);

            // boudingボックスを設定
            {
                float tmp_box_max[3] = { 0.0f, 0.0f, 0.0f };
                float tmp_box_min[3] = { 0.0f, 0.0f, 0.0f };
                auto& position_buffer = patch.buffers.at(mesh.vertex_buffer_id[AttributeType::Position]);

                auto value_type = position_buffer.value_type;
                auto element_type = position_buffer.element_type;
                size_t byte_stride = CalculateByteStride(value_type, element_type);

                for (size_t k = 0; k < indices.size(); k++)
                {
                    uint32_t index = indices.at(k);
                    float position[3];

                    AdapterFloatArrayInterface(position, index, position_buffer.data.data(), byte_stride, value_type, ConvertElementTypeBitSize(element_type));

                    tmp_box_max[0] = std::max(tmp_box_max[0], position[0]);
                    tmp_box_max[1] = std::max(tmp_box_max[1], position[1]);
                    tmp_box_max[2] = std::max(tmp_box_max[2], position[2]);

                    tmp_box_min[0] = std::min(tmp_box_min[0], position[0]);
                    tmp_box_min[1] = std::min(tmp_box_min[1], position[1]);
                    tmp_box_min[2] = std::min(tmp_box_min[2], position[2]);
                }
                mesh.box_max[0] = tmp_box_max[0];
                mesh.box_max[1] = tmp_box_max[1];
                mesh.box_max[2] = tmp_box_max[2];

                mesh.box_min[0] = tmp_box_min[0];
                mesh.box_min[1] = tmp_box_min[1];
                mesh.box_min[2] = tmp_box_min[2];
            }
        }
    }
}

void PerseImpl(Patch& patch, tinyobj::attrib_t& attrib, std::vector < tinyobj::shape_t>& shapes, std::vector<tinyobj::material_t>& materials, std::string filename, std::string filepath)
{
    patch.name = filename;
    patch.models.resize(1);
    patch.samplers.resize(1);
    patch.transforms.resize(1);

    patch.models.at(0).transform_id = 0;
    patch.models.at(0).name = filename;

    PerseAttribute(patch, attrib, filename);
    PerseMaterial(patch, materials, filename);
    PerseTexture(patch, filepath);
    PerseShape(patch, patch.models.at(0), shapes, filename);

}
bool ObjPerser::Perse(Patch& patch, const std::string& input_path)
{
    std::string file_path = input_path;
    std::string file_name = input_path;
    size_t last_back_slush = input_path.find_last_of("\\");
    file_path.replace(last_back_slush, std::string::npos, "\\");
    file_name.replace(0, last_back_slush + 1, "");
    size_t last_dot = file_name.find_last_of(".");
    file_name.replace(last_dot, std::string::npos, "");

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    bool ret = LoadObj(attrib, shapes, materials, input_path, file_path);

    if (ret) {
        PerseImpl(patch, attrib, shapes, materials, file_name, file_path);
    }

    return ret;
}
