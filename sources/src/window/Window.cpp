/**
 * @file    Window.hpp
 * @brief   ウインドウ処理
 */

#include "window/Window.hpp"
#include "window/DX11Window.hpp"

namespace slug
{
#if ENABLE_WINDOW
    
    /**
     * @brief ウインドウ処理の実態クラス
    */
    class Window::Implement
    {
    public:
        /**
         * @brief 初期化、生成を行う
         * @param param ウインドウパラメータ
         * @return 生成に成功
        */
        bool Initialize(const WindowParam& param)
        {
            if (param.type == WindowType::DX11)
            {
                auto window = std::make_shared<DX11Window>();
                m_native_window = std::dynamic_pointer_cast<IWindow>(window);
            }
            else
            {
                return false;
            }

            if (!m_native_window)
            {
                return false;
            }

            return m_native_window->Initialize(param.width, param.height, param.title.c_str());
        }

        /**
         * @brief 更新
         * @param output 描画するバッファ情報
        */
        void Update(const ImageBuffer& output)
        {
            if (m_native_window)
            {
                m_native_window->Update(output);
            }
        }

        /**
         * @brief 終了
        */
        void Terminate()
        {
            if (m_native_window)
            {
                m_native_window->Terminate();
            }
        }

        /**
         * @brief ウインドウを閉じるかチェック
        */
        bool CheckShouldClose()
        {
            if (m_native_window) {
                return m_native_window->CheckShouldClose();
            }
            return false;
        }
    private:
        std::shared_ptr<IWindow> m_native_window;
    };

    Window::Window()
        :m_impl(new Implement)
    {
    }

    Window::~Window()
    {

    }

    bool Window::Create(const WindowParam& param)
    {
        m_param = param;
        return m_impl->Initialize(param);
    }

    void Window::Update(const ImageBuffer& output)
    {
        m_impl->Update(output);
    }

    void Window::Terminate()
    {
        m_impl->Terminate();
    }

    bool Window::CheckShouldClose()
    {
        return m_impl->CheckShouldClose();
    }
#else
class Window::Implement 
{
};
Window::Window()
    :m_impl(new Implement)
{
}

Window::~Window()
{

}

bool Window::Create(const WindowParam&)
{
    return false;
}

void Window::Update(const ImageBuffer&)
{
}

void Window::Terminate()
{
}

bool Window::CheckShouldClose()
{
    return false;
}
#endif // ENABLE_WINDOW
} // namespace slug