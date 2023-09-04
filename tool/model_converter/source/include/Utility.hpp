/**
 * @file    Serialize.hpp
 * @brief   シリアライズ処理のヘッダー
 */

#pragma once

#include "ModelPatch.hpp"

bool CreateOutputDir(const std::string& output_path);

bool ConvertTexture(Patch& patch, const std::string& output_path);

bool OutputBinary(Patch& patch,const std::string& output_path);

bool OutputJson(Patch& patch,const std::string& output_path);

bool InputBinary(Patch& patch,const std::string& output_path);

bool InputJson(Patch& patch,const std::string& output_path);

void AdapterFloatArrayInterface(float* value, const uint32_t index, const uint8_t* dataAddress, const size_t byteStride, ValueType value_type, size_t element_size);

size_t ConvertValueTypeBitSize(ValueType value_type);

size_t ConvertElementTypeBitSize(ElementType element_type);

size_t CalculateByteStride(ValueType value_type, ElementType element_type);
