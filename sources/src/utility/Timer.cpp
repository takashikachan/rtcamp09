/**
* @file    Timer.cpp
* @brief   簡易タイマークラスのソースファイル
*/

#include<time.h>

#include "utility/Timer.hpp"

namespace slug
{
Timer::Timer(float* time)
{
    if (time && !m_time)
    {
        clock_t start = clock();
        m_start_time = (float)start;
        m_time = time;
        *m_time = 0.0f;
    }
}

Timer::~Timer()
{
    if (m_time)
    {
        clock_t end = clock();
        m_end_time = (float)end;
        *m_time = ((m_end_time - m_start_time) / CLOCKS_PER_SEC);
    }
}
} // namespace slug