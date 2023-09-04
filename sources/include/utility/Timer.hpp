/**
 * @file    Timer.hpp
 * @brief   簡易タイマークラスの定義ファイル
 */
#pragma once

#define WATCH_PROCESS_TIME_PREFIX(time, prefix) slug::Timer _timer_##prefix(&time);

namespace slug
{

/**
 * @brief 簡易タイマー
 * @detail 秒を計測します。
*/
class Timer {
public:
    /**
     * @brief コンストラクタ
    */
    Timer(float* time);

    /**
     * @brief デストラクタ
    */
    ~Timer();
private:
    float* m_time = nullptr;      //!< 保存しておくポインタ
    float m_start_time = 0.0f;    //!< 計測開始時間
    float m_end_time = 0.0f;      //!< 計測終了時間
};
} // namespace slug