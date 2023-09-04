/**
 * @file    DX11Window.hpp
 * @brief   DX11のウインドウ処理
 */

#pragma once

#include "window/Window.hpp"
#if ENABLE_WINDOW

namespace slug
{
/**
 * @brief DX11のウインドウクラス
*/
class DX11Window : public IWindow
{
public:
    /**
     * @brief コンストラクタ
    */
    DX11Window();

    /**
     * @brief デストラクタ
    */
    ~DX11Window();

    /**
     * @brief 初期化
     * @param width ウインドウ横サイズ
     * @param height ウインドウ縦サイズ
     * @param title ウインドウタイトル名
     * @return 成功か
    */
    bool Initialize(uint32_t width, uint32_t height, const char* title);

    /**
     * @brief 更新
     * @param output 更新する画像データ
    */
    void Update(const ImageBuffer& output);
    
    /**
     * @brief 終了
    */
    void Terminate();

    /**
     * @brief 閉じるべきか判定
     * @return trueなら、ウインドウを閉じて終了すべき
    */
    bool CheckShouldClose();
private:
    // 内部実装
    struct Implement;
    std::unique_ptr<Implement> m_impl = nullptr;
};
} // namespace slug
#endif // ENABLE_WINDOW