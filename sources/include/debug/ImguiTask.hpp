/**
 * @file    ImguiTask.hpp
 * @brief   Imguiによるデバッグ処理
 */

#pragma once

#if MODE_DEBUG
#include <memory>

#include "Application.hpp"

namespace slug
{
/**
 * @brief Imguiによるデバッグ処理
*/
class ImguiTask
{
public:
    /**
     * @brief コンストラクタ
    */
    ImguiTask();

    /**
     * @brief デストラクタ
    */
    virtual ~ImguiTask();

    /**
     * @brief 初期化
     * @return 成功か
    */
    bool Initialzie();

    /**
     * @brief 更新
     * @param manager グラフィックスマネージャー
     * @param time_config 時間情報
    */
    void Update(GraphicsManager& manager, ApplicationTimeConfig& time_config);

    /**
     * @brief 終了
    */
    void Terminate();
private:
    struct Implement;
    std::unique_ptr<Implement> m_impl;
};
} // namespace slug
#endif MODE_DEBUG