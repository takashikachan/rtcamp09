/**
 * @file    SystemPath.hpp
 * @brief   システムパスの取得処理
 */

#pragma once

#include <string>

namespace slug
{

/**
 * @brief ルートパスを取得
*/
extern std::string GetDirectoryWithRoot();

/**
 * @brief exeが配置しているパスを取得
*/
extern std::string GetDirectoryWithExecutable();

/**
 * @brief packageフォルダを配置しているパスを取得
*/
extern std::string GetDirectoryWithPackage();

/**
 * @brief resouceフォルダを配置しているパスを取得
*/
extern std::string GetDirectoryWithResouce();
} // namespace slug