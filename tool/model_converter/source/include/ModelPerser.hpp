/**
 * @file    ModelPerser.hpp
 * @brief   モデルのパースを行うインターフェースのヘッダ
 *
 */

#pragma once


#include "ModelPatch.hpp"
#include <sstream>
#define ASSERT_MSG(expression, text) \
{ \
    std::stringstream _strstream = {};\
    _strstream << text; \
    bool is_assert = static_cast<bool>(expression);\
    assert(is_assert&& _strstream.str().c_str());\
    if(!is_assert) {__debugbreak(); }\
} \

class IModelPerser {
public:
    IModelPerser() = default;
    virtual ~IModelPerser() = default;
    virtual bool Perse(Patch& patch, const std::string& input_path) = 0;
};
