/**
 * @file    DX11Window.cpp
 * @brief   DX11のウインドウ処理
 */

#pragma once

#include "window/DX11Window.hpp"

#if ENABLE_WINDOW
#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_6.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_win32.h>
#include <imgui/imgui_impl_dx11.h>
#include <unordered_map>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib ")

// imgui用のコールバックを前方宣言
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace slug
{
/**
 * @brief ウインドウコールバック
*/
LRESULT WindowProcedure(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
    if (msg == WM_DESTROY)
    {
        PostQuitMessage(0);
        return 0;
    }

    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam)) 
    {
        return true;
    }

    return DefWindowProc(hwnd, msg, wparam, lparam);
}

/**
 * @brief DX11用ウインドウ
*/
struct DX11Window::Implement
{
public:
    /**
     * @brief ウインドウ生成
    */
    bool CreateDX11Window(uint32_t width, uint32_t height, const char* title)
    {
        //ウィンドウクラス生成＆登録
        WNDCLASSEX w = {};
        w.cbSize = sizeof(WNDCLASSEX);
        w.lpfnWndProc = (WNDPROC)WindowProcedure;
        w.lpszClassName = title;
        w.hInstance = ::GetModuleHandle(0);
        ::RegisterClassEx(&w);

        RECT wrc = { 0,0, (LONG)width, (LONG)height };
        AdjustWindowRect(&wrc, WS_OVERLAPPEDWINDOW, false);
        hwnd_impl = ::CreateWindow(
            w.lpszClassName,
            title,
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            wrc.right - wrc.left,
            wrc.bottom - wrc.top,
            nullptr,
            nullptr,
            w.hInstance,
            nullptr);
        ShowWindow(hwnd_impl, SW_SHOWNORMAL);
        SetWindowLongPtr(hwnd_impl, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
        DragAcceptFiles(hwnd_impl, TRUE);
        ShowCursor(FALSE);
        return true;
    }

    /**
     * @brief DX11のデバイス生成
    */
    bool CreateDevice() 
    {
        D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
        UINT creationFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(DEBUG_BUILD)
        creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

        HRESULT hResult = D3D11CreateDevice(0, D3D_DRIVER_TYPE_HARDWARE,
            0, creationFlags,
            featureLevels, ARRAYSIZE(featureLevels),
            D3D11_SDK_VERSION, &d3d11_device,
            0, &d3d11_device_context);

        return hResult >= 0;
    }

    /**
     * @brief スワップチェーン生成
    */
    bool CreateSwapchain(int32_t width, int32_t height) 
    {
        // Get DXGI Factory (needed to create Swap Chain)
        IDXGIFactory2* dxgiFactory;
        {
            IDXGIDevice1* dxgiDevice;
            HRESULT hResult = d3d11_device->QueryInterface(__uuidof(IDXGIDevice1), (void**)&dxgiDevice);
            assert(SUCCEEDED(hResult));

            IDXGIAdapter* dxgiAdapter;
            hResult = dxgiDevice->GetAdapter(&dxgiAdapter);
            assert(SUCCEEDED(hResult));
            dxgiDevice->Release();

            DXGI_ADAPTER_DESC adapterDesc;
            dxgiAdapter->GetDesc(&adapterDesc);

            OutputDebugStringA("Graphics Device: ");
            OutputDebugStringW(adapterDesc.Description);

            hResult = dxgiAdapter->GetParent(__uuidof(IDXGIFactory2), (void**)&dxgiFactory);
            assert(SUCCEEDED(hResult));
            dxgiAdapter->Release();
        }

        DXGI_SWAP_CHAIN_DESC1 d3d11SwapChainDesc = {};
        d3d11SwapChainDesc.Width = width;
        d3d11SwapChainDesc.Height = height;
        d3d11SwapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        d3d11SwapChainDesc.SampleDesc.Count = 1;
        d3d11SwapChainDesc.SampleDesc.Quality = 0;
        d3d11SwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        d3d11SwapChainDesc.BufferCount = 2;
        d3d11SwapChainDesc.Scaling = DXGI_SCALING_STRETCH;
        d3d11SwapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
        d3d11SwapChainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
        d3d11SwapChainDesc.Flags = 0;

        HRESULT hResult = dxgiFactory->CreateSwapChainForHwnd(d3d11_device, hwnd_impl, &d3d11SwapChainDesc, 0, 0, &d3d11_swap_chain);
        assert(SUCCEEDED(hResult));

        dxgiFactory->Release();
        return true;
    }

    /**
     * @brief バックバッファ用のRenderTargetrViewを生成
    */
    bool CreateBackbufferView() 
    {
        ID3D11Texture2D* d3d11FrameBuffer;
        HRESULT hResult = d3d11_swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&d3d11FrameBuffer);
        assert(SUCCEEDED(hResult));

        hResult = d3d11_device->CreateRenderTargetView(d3d11FrameBuffer, 0, &d3d11_framebuffer_view);
        assert(SUCCEEDED(hResult));
        d3d11FrameBuffer->Release();
        return true;
    }

    /**
     * @brief imguiコンテキストを生成
    */
    bool CreateImgui()
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        io.MouseDrawCursor = true;
        ImGui::StyleColorsDark();

        ImGui_ImplWin32_Init(hwnd_impl);
        ImGui_ImplDX11_Init(d3d11_device, d3d11_device_context);
        return true;
    }

    /**
     * @brief 描画内容を更新
    */
    void Update(const ImageBuffer& output)
    {
        // 黒色でクリア
        FLOAT backgroundColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        d3d11_device_context->ClearRenderTargetView(d3d11_framebuffer_view, backgroundColor);

        // imageBufferの内容をコピー
        ID3D11Resource* resource = {};
        d3d11_framebuffer_view->GetResource(&resource);
        uint32_t row_pitch = output.width * 4 * sizeof(uint8_t);
        d3d11_device_context->UpdateSubresource(resource, 0, nullptr, output.data, row_pitch, 0);

        // imguiを描画
        ImGui::Render();
        d3d11_device_context->OMSetRenderTargets(1, &d3d11_framebuffer_view, nullptr);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        d3d11_swap_chain->Present(1, 0);
    }

    /**
     * @brief 終了処理
    */
    void Terminate() 
    {
        // imguiを終了
        ImGui_ImplDX11_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();

        // dx11を終了
        d3d11_framebuffer_view->Release();
        d3d11_swap_chain->Release();
        d3d11_device_context->Release();
        d3d11_device->Release();

        // ウインドウを削除
        ::DestroyWindow(hwnd_impl);
    }

    /**
     * @brief 終了すべきか判定する処理
    */
    bool CheckShouldClose() 
    {
        if (::PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) 
        {
            TranslateMessage(&msg);
            ::DispatchMessage(&msg);
        }

        return (msg.message == WM_QUIT);
    }
private:
    HWND hwnd_impl = 0;
    MSG msg = {};
    ID3D11Device* d3d11_device = nullptr;
    ID3D11DeviceContext* d3d11_device_context = nullptr;
    IDXGISwapChain1* d3d11_swap_chain = nullptr;
    ID3D11RenderTargetView* d3d11_framebuffer_view = nullptr;
};

// コンストラクタ
DX11Window::DX11Window()
    :m_impl(new Implement)
{
}

// デストラクタ
DX11Window::~DX11Window() 
{
    
}

// 初期化
bool DX11Window::Initialize(uint32_t width, uint32_t height, const char* title) 
{
    
    if (!m_impl->CreateDX11Window(width, height, title)) 
    {
        return false;
    }
    
    if (!m_impl->CreateDevice()) 
    {
        return false;
    }
    
    if (!m_impl->CreateSwapchain(width, height)) 
    {
        return false;
    }
    if (!m_impl->CreateBackbufferView()) 
    {
        return false;
    }

    if (!m_impl->CreateImgui())
    {
        return false;
    }
    return true;
}

// 更新
void DX11Window::Update(const ImageBuffer& output) 
{
    m_impl->Update(output);
}

// 終了
void DX11Window::Terminate() 
{
    m_impl->Terminate();
}

// 閉じるべきか判定
bool DX11Window::CheckShouldClose() 
{
    m_should_close = m_impl->CheckShouldClose();
    return m_should_close;
}
} // namespace slug
#endif // ENABLE_WINDOW