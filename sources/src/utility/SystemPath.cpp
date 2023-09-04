/**
 * @file    SystemPath.hpp
 * @brief   システムパスの取得処理
 */

#include <Windows.h>
#include <filesystem>

#include "utility/SystemPath.hpp"

namespace slug
{

/**
 * @brief ルートパスを探す(.rootが配置しているパス)
 * @param current_path 現在のパス
 * @param root_path ルートパス
*/
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
    }
    else {
        SearchRootPath(current_path.parent_path(), root_path);
    }
}

std::string GetDirectoryWithRoot()
{
    std::filesystem::path result = std::filesystem::current_path();
    
    std::string root_path = {};
    SearchRootPath(result, root_path);
    return root_path;
}

std::string GetDirectoryWithExecutable()
{

    char path[260] = { 0 };
    if (GetModuleFileNameA(nullptr, path, _countof(path)) == 0) {
        return "";
    }

    std::filesystem::path result = path;
    result = result.parent_path();

    return result.string();
}

std::string GetDirectoryWithPackage()
{

    std::string result = GetDirectoryWithRoot();
    result = result + "\\package";
    return result;
}

std::string GetDirectoryWithResouce()
{

    std::string result = GetDirectoryWithRoot();
    result = result + "\\resouce";
    return result;
}
}// namespace slug