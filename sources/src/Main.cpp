/**
 * @file    Main.cpp
 * @brief   エントリポイント
 */

#include <windows.h>
#include <iostream>
#include "Application.hpp"
#include "utility/Timer.hpp"
#include "utility/FileSystem.hpp"

 /**
 * @brief コマンドライン情報
*/
struct ArgumentConfig
{
    std::string title = "";
    std::string output_path = ".\\";
    std::string width = "512";
    std::string height = "512";
    std::string time = "10";
    std::string framerate = "60";
    bool use_optix = false;
    bool enable_saveimage = false;
    bool enable_window = true;
    bool enable_debug = false;
};

/*
* @brief コマンドライン引き数を解析
* @param[in] argc コマンドライン引き数の数
* @param[in] argv コマンドライン引き数
* @param[in] result 解析したコマンドライン引き数情報
*/
void ParseArgument(int argc, char** argv, ArgumentConfig& result)
{
    for (auto i = 0; i < argc; ++i)
    {
        if (_stricmp(argv[i], "-o") == 0)
        {
            i++;
            result.output_path = argv[i];
        }
        else if (_stricmp(argv[i], "-width") == 0)
        {
            i++;
            result.width = argv[i];
        }
        else if (_stricmp(argv[i], "-height") == 0)
        {
            i++;
            result.height = argv[i];
        }
        else if (_stricmp(argv[i], "-title") == 0)
        {
            i++;
            result.title = argv[i];
        }
        else if (_stricmp(argv[i], "-time") == 0)
        {
            i++;
            result.time = argv[i];
        }
        else if (_stricmp(argv[i], "-framerate") == 0)
        {
            i++;
            result.framerate = argv[i];
        }
        else if (_stricmp(argv[i], "-optix") == 0)
        {
            result.use_optix = true;
        }
        else if (_stricmp(argv[i], "-saveimage") == 0)
        {
            result.enable_saveimage = true;
        }
        else if (_stricmp(argv[i], "-window") == 0)
        {
            result.enable_window = true;
        }
        else if (_stricmp(argv[i], "-debug") == 0)
        {
            result.enable_debug = true;
        }
    }
}

/**
 * @brief 時間計測をテキスト出力
*/
void WriteTxtTimerCondig(const slug::ApplicationTimeConfig& time, const float& total, const std::string& output_path) 
{
    slug::DefaultFileSystem filesystem = {};
#if ENABLE_WRITE_PERFORMANCE_LOG
    // パフォーマンステキストを出力
    {
        std::string output_file = output_path + "\\" + "performance.txt";
        char str[512];
        
        sprintf_s(str,"total:%f(s)\n, framecount:%d\n, update:%f(s)\n, draw:%f(s)\n, save:%f(s)\n, ave_update:%f(s)\n, ave_draw:%f\n, ave_save:%f(s)\n",
            total, time.framecount, time.total_update_time, time.total_draw_time, time.total_save_time, time.average_update_time, time.average_draw_time, time.average_save_time
        );

        std::string stri = str;
        filesystem.WriteFile(output_file, stri.data(), stri.size());
    }
#endif
    {
        std::string output_file = output_path + "\\" + "fps.txt";
        char str[512];
        uint32_t fps = time.framecount / static_cast<uint32_t>(time.rendering_time);
        sprintf_s(str, "%d", fps);
        std::string stri = str;
        filesystem.WriteFile(output_file, stri.data(), stri.size());
    }
}

/**
 * @brief エントリポイント
*/
int main(int argc, char** argv)
{
#if !defined(MODE_DEBUG)
    FreeConsole();
#endif
    slug::ApplicationTimeConfig time_config = {};
    slug::ApplicationParam app_param = {};
    slug::ReturnCode ret = slug::ReturnCode::Success;

    float total_process_time = 0.0f;
    {
        WATCH_PROCESS_TIME_PREFIX(total_process_time, total_process);

        // コマンドライン引数からアプリケーションパラメータを抽出
        if (argc > 1)
        {
            ArgumentConfig arg_config = {};
            ParseArgument(argc, argv, arg_config);

            if (arg_config.use_optix)
            {
                app_param.api = slug::GraphicsAPI::Optix;
            }
            else
            {
                app_param.api = slug::GraphicsAPI::Invalid;
            }

            app_param.title = "";
            app_param.output_path = ".\\";
            app_param.enable_debug = arg_config.enable_debug;
            app_param.enable_window = arg_config.enable_window;
            app_param.enable_saveimage = arg_config.enable_saveimage;
            app_param.width = std::stoi(arg_config.width);
            app_param.height = std::stoi(arg_config.height);
            app_param.rendering_time = std::stoi(arg_config.time);
            app_param.framerate = std::stoi(arg_config.framerate);
        }
        else
        {
            app_param.api = slug::GraphicsAPI::Optix;
            app_param.output_path = ".\\";
            app_param.title = "";
            app_param.width = 600;
            app_param.height = 600;
            app_param.rendering_time = 8;
            app_param.framerate = 16;
            app_param.enable_saveimage = false;
            app_param.enable_debug = false;
            app_param.enable_window = true;
        }

        if (app_param.title.empty())
        {
            app_param.title = "slug_raytracing";
        }

        // アプリケーションを実行
        slug::Application app(app_param);

        ret = app.Run();
        // 終了した場合、時間計測情報を取得
        time_config = app.GetTimeConfig();
    }

    if (!app_param.enable_window)
    {
        // 時間計測系をtxt出力する。
        WriteTxtTimerCondig(time_config, total_process_time, app_param.output_path);
    }
    return static_cast<int>(ret);
}
