/**
 * @file    GltfPerser.hpp
 * @brief   Gltfのパースを行うクラスのヘッダ
 */

#pragma once

#include "ModelPerser.hpp"

class GltfPerser : public IModelPerser {
public:
    GltfPerser();
    virtual ~GltfPerser();
    bool Perse(Patch& patch, const std::string& input_path);
};
