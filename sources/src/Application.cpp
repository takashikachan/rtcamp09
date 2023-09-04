/**
 * @file    Application.hpp
 * @brief   アプリケーションクラスの定義ファイル
 */

#include "utility/Timer.hpp"
#include "utility/Define.hpp"
#include "utility/FileSystem.hpp"
#include "Application.hpp"
#include "graphic/ImageBuffer.hpp"

#if MODE_DEBUG
#include "debug/ImguiTask.hpp"
#endif 

namespace slug
{
#if MODE_DEBUG
static ImguiTask g_imgui_task;
#endif

Application::Application(const ApplicationParam& param) 
    :m_param(param)
{
    m_time_config.rendering_time = param.rendering_time;
}

Application::~Application() 
{

}

ReturnCode Application::Run()
{
    if (!m_initialzed) 
    {
        if (!Initialize())
        {
            return ReturnCode::InitializeFailed;
        }
        m_initialzed = true;
    }

    bool should_exit = false;
    while (!should_exit) 
    {
        should_exit = Update();
    }

    if (!Terminate()) 
    {
        return ReturnCode::TerminateFailed;
    }

    return ReturnCode::Success;

}

const ApplicationParam& Application::GetParam() const 
{
    return m_param;
}

const ApplicationTimeConfig& Application::GetTimeConfig() const
{
    return m_time_config;
}

bool Application::Initialize() 
{
    if(m_param.enable_window)
    {
        WindowParam window_param = {};
        window_param.width = m_param.width;
        window_param.height = m_param.height;
        window_param.title = m_param.title;
        window_param.type = WindowType::DX11;
        m_param.enable_window = m_window.Create(window_param);

#if MODE_DEBUG
        g_imgui_task.Initialzie();
#endif
    }
    
    GraphicsManagerParam gfx_param = {};
    gfx_param.render_width = m_param.width;
    gfx_param.render_height = m_param.height;
    gfx_param.enable_debug = m_param.enable_debug;
    m_graphics_manager.Initialize(gfx_param);

    return true;
}

bool Application::Update() 
{
    {
        WATCH_PROCESS_TIME_PREFIX(m_time_config.update_time, update);

        ImageBuffer buffer = {};
        {
            WATCH_PROCESS_TIME_PREFIX(m_time_config.draw_time, draw);
            m_graphics_manager.Update(m_time_config.framecount);
            m_graphics_manager.GetOutputBuffer(buffer);
        }

        // データ保存
        if (m_param.enable_saveimage && m_time_config.framecount < m_param.framerate * m_param.rendering_time)
        {
            WATCH_PROCESS_TIME_PREFIX(m_time_config.save_time, save);
#if 1
            // JPG画像保存
            std::string output_path = m_param.output_path;
            output_path = output_path;

            char str[1024];
            sprintf_s(str, "%s%03d.png", output_path.c_str(), m_time_config.framecount);
            SaveImageJPG(buffer, str);
#else
            // バイナリデータ保存
            DefaultFileSystem filesystem = {};
            std::string output_path = m_param.output_path;
            output_path = output_path + "\\tmp\\";

            char str[1024];
            sprintf_s(str, "%s%03d.image", output_path.c_str(), m_time_config.framecount);
            filesystem.WriteFile(str, buffer.data, buffer.GetByteSize());
#endif
        }

        // ウインドウ描画
        if(m_param.enable_window)
        {
#if MODE_DEBUG
            g_imgui_task.Update(m_graphics_manager, m_time_config);
#endif

            m_window.Update(buffer);
        }
    }

    // 処理時間を計測
    CalculateTimestamp();

    // 終了するべきか
    return CheckShouldExit();
}

bool Application::Terminate() 
{
    if (m_param.enable_window)
    {
#if MODE_DEBUG
        g_imgui_task.Terminate();
#endif
        m_window.Terminate();
    }
    m_graphics_manager.Terminate();
    return true;
}

void Application::CalculateTimestamp()
{
    m_time_config.total_update_time += m_time_config.update_time;
    m_time_config.total_draw_time += m_time_config.draw_time;
    m_time_config.total_save_time += m_time_config.save_time;
    m_time_config.framecount++;

#if ENABLE_WRITE_PERFORMANCE_LOG
    // 時間計測
    {
        m_time_config.interval_update_time += m_time_config.update_time;
        m_time_config.interval_draw_time += m_time_config.draw_time;
        m_time_config.interval_save_time += m_time_config.save_time;

        if ((m_time_config.framecount % m_time_config.average_frame_interval) == 0)
        {
            m_time_config.average_update_time = m_time_config.interval_update_time / m_time_config.average_frame_interval;
            m_time_config.average_draw_time = m_time_config.interval_draw_time / m_time_config.average_frame_interval;
            m_time_config.average_save_time = m_time_config.interval_save_time / m_time_config.average_frame_interval;

            m_time_config.average_fps = 1.0f / (m_time_config.average_update_time);

            m_time_config.interval_update_time = 0.0f;
            m_time_config.interval_draw_time = 0.0f;
            m_time_config.interval_save_time = 0.0f;
        }
    }
#endif
}

bool Application::CheckShouldExit()
{
    if (m_param.enable_window)
    {
        return m_window.CheckShouldClose();
    }
    else if (static_cast<uint32_t>(m_time_config.total_update_time) >= m_param.max_rendering_time)
    {
        return true;
    }
    else if (m_time_config.framecount >= (m_param.rendering_time * m_param.framerate)) 
    {
        return true;
    }

    return false;
}
}