/**
 * @file    Window.hpp
 * @brief   ウインドウ処理
 */

#pragma once

#define ENABLE_WINDOW MODE_DEBUG 

#include <cstdint>
#include <memory>
#include <string>
#include "graphic/ImageBuffer.hpp"

namespace slug
{

/**
 * @brief ウインドウの種類
*/
enum class WindowType : uint8_t
{
    DX11 = 0,   //!< DX11用
    Invalid
};

/**
 * @brief ウインドウのパラメータ
*/
struct WindowParam 
{
    WindowType type = WindowType::DX11; //!< ウインドウの種類
    uint32_t width = 0;                 //!< 横サイズ
    uint32_t height = 0;                //!< 縦サイズ
    std::string title = {};             //!< タイトル名
};

/**
 * @brief ウインドウのインターフェース
*/
class IWindow
{
public:
    IWindow() = default;
    virtual ~IWindow() = default;
    virtual bool Initialize(uint32_t width, uint32_t height, const char* title) = 0;
    virtual void Update(const ImageBuffer& output) = 0;
    virtual void Terminate() = 0;
    virtual bool CheckShouldClose() { return m_should_close; }
protected:
    bool m_should_close = false;    //!< 閉じるべきかの判定フラグ
};

class Window 
{
public:
    /**
     * @brief コンストラクタ
    */
    Window();

    /**
     * @brief デストラクタ
    */
    ~Window();

    /**
     * @brief 生成処理
    */
    bool Create(const WindowParam& param);

    /**
     * @brief 更新処理
    */
    void Update(const ImageBuffer& output);

    /**
     * @brief 終了処理
    */
    void Terminate();

    /**
     * @brief 閉じるべきかチェック
    */
    bool CheckShouldClose();
private:
    // 内部実装
    class Implement;
    std::unique_ptr<Implement> m_impl = nullptr;

private:
    WindowParam m_param = {};    //!< ウインドウパラメータ
};
} // namespace slug