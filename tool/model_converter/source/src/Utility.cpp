/**
 * @file    Serialize.cpp
 * @brief   シリアライズ処理のソース
 *
 */

#include "Utility.hpp"
#include <sstream>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <Windows.h>
#include <assert.h>

void SearchRootPath(std::filesystem::path current_path, std::string& root_path)
{
    bool is_found = false;
    for (const auto& file : std::filesystem::directory_iterator(current_path)) {
        if (file.path().filename().string().find(".root") != std::string::npos) {
            is_found = true;
            break;
        }
    }

    if (is_found) {
        root_path = current_path.string();
    } else {
        SearchRootPath(current_path.parent_path(), root_path);
    }
}


bool ExecuteConvertTextureCommand(const std::string& file_path, const std::string& output_path)
{
    std::string root_path;
    std::filesystem::path path = std::filesystem::current_path();
    SearchRootPath(path, root_path);


    std::string command_line = root_path + "\\tool\\tool_executer\\bin\\ToolExecuter.exe";
    command_line = command_line + " -r " + file_path + " -o " + output_path + " -y " + "-f B8G8R8A8_UNORM -m 1";
    int size = MultiByteToWideChar(CP_ACP, 0, command_line.c_str(), -1, (wchar_t*)NULL, 0);
    wchar_t* w_command = new wchar_t[size];
    MultiByteToWideChar(CP_ACP, 0, command_line.c_str(), -1, w_command, size);
    

    STARTUPINFO si {};
    PROCESS_INFORMATION pi {};

    si.cb = sizeof(si);
    if (CreateProcess(nullptr, w_command, nullptr, nullptr, false, 0, nullptr, nullptr, &si, &pi)) {

        // アプリケーション終了まで待つ
        WaitForSingleObject(pi.hProcess, INFINITE);

        unsigned long exitCode;
        GetExitCodeProcess(pi.hProcess, &exitCode);

        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    } else {
        assert(false);
        return false;
    }
    return true;
}

bool CreateOutputDir(const std::string& output_path)
{
    std::filesystem::path path = output_path;
    if (!std::filesystem::exists(path)) {
        return std::filesystem::create_directory(path);
    }
    return true;
}

bool ConvertTexture(Patch& patch, const std::string& output_path)
{
    for (auto& texture : patch.textures) {
        if (ExecuteConvertTextureCommand(texture.path, output_path)) {
            size_t offset = texture.name.find_last_of(".");
            size_t count = texture.name.length() - offset;
            texture.name.replace(offset, count, ".dds");
            texture.path = output_path + "\\" + texture.name;
        }
        else {
            printf_s("Error : Failed ConvertTexture %s \n", texture.path.c_str());
        }
    }
    return true;
}

bool OutputBinary(Patch& patch,const std::string& output_path)
{
    std::string output_file = output_path + "\\" + patch.name + ".model";
    std::ofstream ofs(output_file, std::ios::binary);
    cereal::BinaryOutputArchive output(ofs);
    output(patch);
    return true;
}

bool OutputJson(Patch& patch, const std::string& output_path)
{
    std::string output_file = output_path + "\\" + patch.name + ".json";
    std::ofstream ofs(output_path, std::ios::out);
    cereal::JSONOutputArchive output(ofs);
    output(patch);
    return true;
}

bool InputBinary(Patch& patch, const std::string& output_path)
{
    std::ifstream ifs(output_path, std::ios::binary);
    cereal::BinaryInputArchive input(ifs);
    input(patch);
    return true;
}

bool InputJson(Patch& patch, const std::string& output_path)
{
    Shader shader;
    std::ifstream ifs(output_path, std::ios::in);
    cereal::JSONInputArchive input(ifs);
    input(patch);
    return true;
}

template<typename T, typename U>
void AdapterFloatArray(T* value, const uint32_t index, const uint8_t* dataAddress, const size_t byteStride, size_t element_size)
{
    const U* data = (reinterpret_cast<const U*>(dataAddress + index * byteStride));
    for (size_t i = 0; i < element_size; i++)
    {
        value[i] = static_cast<T>(data[i]);
    }

    if (element_size == 4)
    {
        value[element_size - 1] = static_cast<T>(1.0f);
    }
}

void AdapterFloatArrayInterface(float* value, const uint32_t index, const uint8_t* dataAddress, const size_t byteStride, ValueType value_type, size_t element_size)
{
    if (value_type == ValueType::Float)
    {
        AdapterFloatArray<float, float>(value, index, dataAddress, byteStride, element_size);
    } else
    {
        AdapterFloatArray<float, float>(value, index, dataAddress, byteStride, element_size);
    }
}

size_t ConvertValueTypeBitSize(ValueType value_type)
{
    switch (value_type)
    {
    case ValueType::Uint8:
        return sizeof(uint8_t);
        break;
    case ValueType::Sint8:
        return sizeof(int8_t);
        break;
    case ValueType::Uint16:
        return sizeof(uint16_t);
        break;
    case ValueType::Sint16:
        return sizeof(int16_t);
        break;
    case ValueType::Uint32:
        return sizeof(uint32_t);
        break;
    case ValueType::Sint32:
        return sizeof(int32_t);
        break;
    case ValueType::Float:
        return sizeof(float);
        break;
    case ValueType::Double:
        return sizeof(double);
        break;
    case ValueType::LDouble:
        return sizeof(long double);
        break;
    default:
        return sizeof(uint32_t);
    }
}

size_t ConvertElementTypeBitSize(ElementType element_type)
{
    switch (element_type)
    {
    case ElementType::Vector1:
        return 1;
        break;
    case ElementType::Vector2:
        return 2;
        break;
    case ElementType::Vector3:
        return 3;
        break;
    case ElementType::Vector4:
        return 4;
        break;
    case ElementType::Matrix2x2:
        return 4;
        break;
    case ElementType::Matrix3x3:
        return 9;
        break;
    case ElementType::Matrix4x4:
        return 16;
        break;
    case ElementType::Scalar:
        return 1;
        break;
    case ElementType::Vector:
        return 4;
        break;
    case ElementType::Matrix:
        return 16;
        break;
    default:
        return 1;
        break;
    }
}


size_t CalculateByteStride(ValueType value_type, ElementType element_type)
{
    return ConvertValueTypeBitSize(value_type) * ConvertElementTypeBitSize(element_type);
}
