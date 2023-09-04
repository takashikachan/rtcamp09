/**
 * @file    ObjPerser.hpp
 * @brief   Objのパースを行うクラスのヘッダ
 */

#pragma once

#include "ModelPerser.hpp"

class ObjPerser : public IModelPerser {
public:
    ObjPerser();
    virtual ~ObjPerser();
    bool Perse(Patch& patch, const std::string& input_path);
};
