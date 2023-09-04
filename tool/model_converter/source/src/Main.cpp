/**
 * @file    Main.cpp
 * @brief   エントリーポイント
 *
 */

#include <cstdint>
#include <stdio.h>
#include <string>
#include "GltfPerser.hpp"
#include "ObjPerser.hpp"
#include "Utility.hpp"

enum class ModelPrefixType : uint32_t{
    Obj = 0,
    Fbx,
    Gltf,
    USD,
    Invalid
};

struct Argument {
    std::string input_path = "";
    std::string tmp_dir = "";
    std::string output_dir = "";
};

void PerseArg (int argc, char** argv, Argument& arg){
    for (auto i = 0; i < argc; ++i) {
        if (_stricmp(argv[i], "-i") == 0) {
            i++;
            arg.input_path = argv[i];
        }

        if (_stricmp(argv[i], "-o") == 0) {
            i++;
            arg.output_dir = argv[i];
        }
    }
}

ModelPrefixType DetectModelPrefixType (const std::string& input_path){
    if (input_path.find(".obj") != std::string::npos) {
        return ModelPrefixType::Obj;
    }
    else if(input_path.find(".fbx") != std::string::npos) {
        return ModelPrefixType::Fbx;
    }
    else if (input_path.find(".gltf") != std::string::npos) {
        return ModelPrefixType::Gltf;
    }
    else if (input_path.find(".usd") != std::string::npos) {
        return ModelPrefixType::USD;
    }
    return ModelPrefixType::Invalid;
}

int main(int argc, char** argv)
{
    if (argc <= 1) {
        printf_s("ModelConverter.exe -i [input_path] -o [output_dir]\n");
    }

    Argument arg = {};
    PerseArg(argc, argv, arg);
    if (arg.input_path.empty() || arg.output_dir.empty()) {
        printf_s("Error : Invlid Argument\n");
    }

    printf_s("Execute : ModelConverter.exe -i %s -o %s \n", arg.input_path.c_str(), arg.output_dir.c_str());

    ModelPrefixType prefix_type = DetectModelPrefixType(arg.input_path);
    std::unique_ptr<IModelPerser> perser = nullptr;
    switch (prefix_type) {
    case ModelPrefixType::Gltf:
        perser = std::make_unique<GltfPerser>();
        break;
    case ModelPrefixType::Obj:
        perser = std::make_unique<ObjPerser>();
        break;
    default:
        printf_s("Errror : Not Implement This Model Format\n");
        return -1;
        break;
    }

    Patch patch = {};
    bool ret = false;
    if (perser) {
        printf_s("Perse Object...\n");
        ret = perser->Perse(patch, arg.input_path);
    }

    if (!ret) {
        printf_s("Errror : Perse Error\n");
        return -1;
    }

    printf_s("MakeDir %s \n", arg.output_dir.c_str());
    if (!CreateOutputDir(arg.output_dir)) {
        printf_s("Errror : Cant Create OutputDir %s \n", arg.output_dir.c_str());
        return -1;
    }

    ConvertTexture(patch, arg.output_dir);
    OutputBinary(patch, arg.output_dir);

    return 0;
}
