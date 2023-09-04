/**
 * @file    GraphicsManager.hpp
 * @brief   グラフィックス処理の管理クラス
 */

#pragma once

#include <memory>
#include "graphic/ImageBuffer.hpp"
#include "graphic/CudaPixelBuffer.hpp"
#include "graphic/GraphicsContext.hpp"
#include "utility/Camera.hpp"
#include "scene/SampleScene.hpp"

namespace slug
{
    struct GraphicsManagerParam
    {
        uint32_t render_width = 0;
        uint32_t render_height = 0;
        bool enable_debug = false;
    };

    /**
     * @brief グラフィックス処理の管理クラス
    */
    class GraphicsManager
    {
    public:
        GraphicsManager();
        virtual ~GraphicsManager();
        bool Initialize(const GraphicsManagerParam& param);
        void Update(uint32_t framecount);
        void Terminate();
        void GetOutputBuffer(ImageBuffer& output);
        HdrPixelBuffer& GetHdrBuffer() 
        {
            return m_hdr_buffer;
        }
        Trackball& GetTrackballCamera();
        InitParam& GetInitParam();
    private:
        GraphicsManagerParam m_param = {};
        GraphicsContext m_context = {};
        SampleScene m_scene = {};
        SdrPixelBuffer m_output_buffer;
        HdrPixelBuffer m_hdr_buffer;
        Trackball m_traclball_camera = {};
        InitParam m_init_param = {};
    };
} // namespace slug