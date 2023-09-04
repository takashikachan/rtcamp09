/**
 * @file    Application.hpp
 * @brief   アプリケーションクラスの定義ファイル
 */

#pragma once

#define ENABLE_WRITE_PERFORMANCE_LOG 1//MODE_DEBUG

#include <cstdint>
#include <string>

#include "graphic/GraphicManager.hpp"
#include "window/Window.hpp"

namespace slug
{

/**
 * @brief 使用するAPIの種類
*/
enum class GraphicsAPI : uint8_t
{
    Optix = 0,  //!< Optix
    // Cuda,    //!< Cuda
    // CPU,     //!< CPU
    // DXR,     //!< DXR
    // Vulkan   //!< Vulkan
    // Embree   //!< Embree
    Invalid
};

/**
 * @brief 戻り値
*/
enum class ReturnCode : int8_t
{
    Success = 0,
    Timeout = -1,
    InitializeFailed = -2,
    UpdateFailed = -3,
    TerminateFailed = -4
};

/**
 * @brief アプリケーションの時間計測データ
*/
struct ApplicationTimeConfig
{
    uint32_t framecount = 0;                 //!< フレーム数
    uint32_t rendering_time = 100;
    float total_update_time = 0.0f;          //!< 合計処理時間
    float total_draw_time = 0.0f;            //!< 合計描画時間
    float total_save_time = 0.0f;            //!< 合計保存時間

    float update_time = 0.0f;                //!< 更新処理時間
    float draw_time = 0.0f;                  //!< 描画処理時間
    float save_time = 0.0f;                  //!< 画像保存時間

#if ENABLE_WRITE_PERFORMANCE_LOG
    int32_t average_frame_interval = 5;     //!< 平均時間を算出するフレーム間隔
    float average_update_time = 0.0f;        //!< 平均更新時間
    float average_draw_time = 0.0f;          //!< 平均描画時間
    float average_save_time = 0.0f;          //!< 平均保存時間

    float interval_update_time = 0.0f;       //!< 計測区間の更新時間
    float interval_draw_time = 0.0f;         //!< 計測区間の描画時間
    float interval_save_time = 0.0f;         //!< 計測区間の保存時間

    float average_fps = 0.0f;                //!< fps
#endif
};

/**
 * @brief アプリケーションのパラメータ
*/
struct ApplicationParam
{
    GraphicsAPI api = GraphicsAPI::Optix;   //!< 使用するAPIの種類
    std::string title = "";                 //!< title
    std::string output_path = "";           //!< 出力パス
    uint32_t width = 0;                     //!< 横サイズ
    uint32_t height = 0;                    //!< 縦サイズ
    uint32_t rendering_time = 10;           //!< 描画時間
    uint32_t framerate = 60;                //!< フレームレート
    bool enable_window = false;             //!< ウインドウを使用する
    bool enable_debug = false;              //!< デバッグを有効化する
    bool enable_saveimage = false;          //!< 画像を保存する
    const uint32_t max_rendering_time = 295;//!< 最大描画時間
};

/**
 * @brief アプリケーションクラス
*/
class Application
{
public:
    /**
     * @brief コンストラクタ
    */
    Application(const ApplicationParam& param);

    /**
     * @brief デストラクタ
    */
    ~Application();

    /**
     * @brief 実行
    */
    ReturnCode Run();

    /**
     * @brief パラメータを取得
    */
    const ApplicationParam& GetParam() const;

    /**
     * @brief 時間計測情報を取得
    */
    const ApplicationTimeConfig& GetTimeConfig() const;
private:
    /**
     * @brief 初期化
    */
    bool Initialize();

    /**
     * @brief 更新
    */
    bool Update();

    /**
     * @brief 終了
    */
    bool Terminate();

    /**
     * @brief タイムスタンプを計算
    */
    void CalculateTimestamp();

    /**
     * @brief 終了するべきかチェック
    */
    bool CheckShouldExit();
private:
    ApplicationTimeConfig m_time_config = {}; //!< 時間計測
    ApplicationParam m_param = {};            //!< パラメータ
    GraphicsManager m_graphics_manager = {};  //!< 描画用のコンテキスト
    Window m_window = {};                     //!< ウインドウ処理
    bool m_initialzed = false;                //!< 初期化済か
};
} // namespace slug