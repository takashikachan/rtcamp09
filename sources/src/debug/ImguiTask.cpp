/**
 * @file    ImguiTask.cpp
 * @brief   Imguiによるデバッグ処理
 */

#pragma once

#if MODE_DEBUG
#include <cuda_runtime.h>
#include <optix.h>


#include "graphic/ImageBuffer.hpp"
#include "DirectXTex.h"
#include "debug/ImguiTask.hpp"
#include <imgui/imgui.h>
#include <imgui/imgui_impl_win32.h>
#include <imgui/imgui_impl_dx11.h>

#include "utility/LensSystem.hpp"
#include "utility/SystemPath.hpp"


using namespace std;
#include "optix_slug/random.h"
#include "optix_slug/cmj_utility.h"
#include "optix_slug/default_shader_data.h"

namespace slug
{

struct ImguiTask::Implement
{
    float move_speed = 0.01f;
    bool stop_camera = true;
    float3 debug_rbg_color = {};
    float time = 6.0f;
    ArHosekSkyModelState sky_state = {};
    std::vector<float> solar_data = {};
    int32_t lens_file = 9;
    void Initialize() 
    {
    }

    void Begin()
    {
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Setting");
    }

    void UpdateTimeConfig(ApplicationTimeConfig& time_config)
    {
        ImGui::LabelText("", "Time Config");
        ImGui::SliderInt("Average Frame Count", &time_config.average_frame_interval, 10, 240);
        ImGui::Text("Frame Count  : %d", time_config.framecount);
        ImGui::Text("Fps          : %f (ms)", time_config.average_fps);
        ImGui::Text("Update Time   : %f (ms)", time_config.average_update_time);
        ImGui::Text("Draw Time : %f (ms)", time_config.average_draw_time);
        ImGui::Text("Save Time  : %f (ms)", time_config.average_save_time);
    }

    void UpdateTrackballCamera(Trackball& track_ball_camera)
    {
        ImGuiIO& io = ImGui::GetIO();

        Camera& camera = track_ball_camera.GetCamera();
        float position[3] = { camera.GetPosition().x, camera.GetPosition().y, camera.GetPosition().z };
        float target[3] = { camera.GetRegardPosition().x, camera.GetRegardPosition().y, camera.GetRegardPosition().z };
        float upvector[3] = { camera.GetUpVector().x, camera.GetUpVector().y, camera.GetUpVector().z };
        float fovy = camera.GetFovy();

        bool dirty = false;
        ImGui::LabelText("", "Camera");
        ImGui::Text("stop camera : %s", stop_camera ? "true" : "false");
        ImGui::Text("positon : %f, %f, %f", position[0], position[1], position[2]);
        ImGui::Text("target : %f, %f, %f", target[0], target[1], target[2]);
        ImGui::Text("upvector : %f, %f, %f", upvector[0], upvector[1], upvector[2]);
        dirty |= ImGui::SliderFloat("move speed", &move_speed, 0.0f, 1.0f);        
        dirty |= ImGui::SliderFloat("fovy", &fovy, 0.0f, 90.0f);
        
        if(ImGui::Button("save_cameara"))
        {
            std::stringstream ss;
            ss << "position : [" << position[0] << "," << position[1] << "," << position[2] << "]\n";
            ss << "target : [" << target[0] << "," << target[1] << "," << target[2] << "]\n";
            printf("%s", ss.str().c_str());
        }

        int32_t x = (int32_t)io.MousePos[0];
        int32_t y = (int32_t)io.MousePos[1];
        int32_t width = (int32_t)ImGui::GetWindowWidth();
        int32_t height = (int32_t)ImGui::GetWindowHeight();
        if (io.MouseDown[1])
        {
            track_ball_camera.SetViewMode(Trackball::LookAtFixed);
            track_ball_camera.UpdateTracking(x, y, width, height);
            dirty |= true;
        }
        else if(io.MouseDown[2])
        {
            track_ball_camera.SetViewMode(Trackball::EyeFixed);
            track_ball_camera.UpdateTracking(x, y, width, height);
            dirty |= true;
        }

        int32_t w = (int32_t)io.MouseWheel;
        if (w != 0)
        {
            track_ball_camera.CalcWheelEvent(w);
            dirty |= true;
        }

        if (ImGui::IsKeyPressed(ImGuiKey_W))
        {
            track_ball_camera.MoveForward(move_speed);
            dirty |= true;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_S))
        {
            track_ball_camera.MoveBackward(move_speed);
            dirty |= true;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_A))
        {
            track_ball_camera.MoveLeft(move_speed);
            dirty |= true;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_D))
        {
            track_ball_camera.MoveRight(move_speed);
            dirty |= true;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_Q))
        {
            track_ball_camera.MoveUp(move_speed);
            dirty |= true;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_E))
        {
            track_ball_camera.MoveDown(move_speed);
            dirty |= true;
        }

        if (ImGui::IsKeyReleased(ImGuiKey_T))
        {
            stop_camera = !stop_camera;
        }

        if (dirty)
        {
            camera.SetFovy(fovy);
            camera.drity = true;
        }
    }

    void UpdateInitParam(InitParam& param) 
    {
        if (ImGui::CollapsingHeader("Launch Param"))
        {
            ImGui::SliderInt("Launch Sample", &param.sample_per_launch, 1, 300);
            ImGui::SliderInt("Max Depth", &param.max_depth, 0, 5);
            ImGui::Checkbox("russian_roulette", &param.russian_roulette);
            ImGui::SliderInt("frame count", &param.framecount, 0, 120);
        }
    }

    void UpdateDebugParam(InitParam& param)
    {
        if (ImGui::CollapsingHeader("Debug Param"))
        {
            int32_t output_type = static_cast<int32_t>(param.output_type);
            ImGui::RadioButton("Output : Albedo", &output_type, (int32_t)OutputType::Albedo);
            ImGui::RadioButton("Output : Normal", &output_type, (int32_t)OutputType::Normal);
            ImGui::RadioButton("Output : Position", &output_type, (int32_t)OutputType::Position);
            ImGui::RadioButton("Output : Hdr", &output_type, (int32_t)OutputType::Hdr);
            ImGui::RadioButton("Output : Sdr", &output_type, (int32_t)OutputType::Sdr);
            param.output_type = (OutputType)output_type;

            ImGui::LabelText("", "Spectrum Param");

            ImGui::RadioButton("Debug Mode : None", &param.debug_mode, DebugMode_None);
            ImGui::RadioButton("Debug Mode : RGB", &param.debug_mode, DebugMode_RGB);
            ImGui::RadioButton("Debug Mode : Spectrum", &param.debug_mode, DebugMode_Spectrum);
            ImGui::RadioButton("Debug Mode : Color", &param.debug_mode, DebugMode_Color);
            ImGui::RadioButton("Debug Mode : Texture", &param.debug_mode, DebugMode_Noise);
            ImGui::RadioButton("Debug Mode : Texture", &param.debug_mode, DebugMode_Texture);


            ImGui::ColorEdit3("Debug RGB", param.debug_color);
        }
    }

    void UpdateSkyParam(InitParam& param)
    {
        if (ImGui::CollapsingHeader("Sky Param"))
        {
            bool dirty = false;
            dirty |= ImGui::SliderFloat("sky intensity", & param.sky_intensity, 0.0f, 1.0f);
            dirty |= ImGui::SliderFloat("atmospheric turbidity", &param.atmospheric_turbidity, 1.0f, 10.0f);
            dirty |= ImGui::SliderFloat("ground albedo", &param.ground_albedo, 0.0f, 1.0f);
            dirty |= ImGui::SliderFloat3("sun dir", param.sun_dir, -1.0f, 1.0f);
            dirty |= ImGui::SliderFloat3("sun emission", param.sun_emission, 0.0f, 10.0f);
            param.sky_dirty = dirty;
        }
    }

    void UpdateLens(LensSystemParam& param, Trackball& track_ball_camera)
    {
        if (ImGui::CollapsingHeader("Lens Param"))
        {
            bool dirty = false;
            dirty |= ImGui::Checkbox("enable lens", &param.enable_lensflare);
            dirty |= ImGui::SliderInt("blade count", &param.blade_count, 1, 10);
            dirty |= ImGui::SliderInt("anamorphic", &param.anamorphic, 1, 10);
            dirty |= ImGui::SliderInt("degree", &param.degree, 1, 10);
            dirty |= ImGui::SliderFloat("sensor_width", &param.sensor_width, 0, 100);
            dirty |= ImGui::SliderFloat("distance", &param.distance, 0.0f, 100.0f);
            dirty |= ImGui::SliderFloat("sample mul", &param.sample_mul, 1.0f, 1000.0f);
            dirty |= ImGui::SliderFloat("r entrance", &param.r_entrance, 0.0f, 30.0f);
            dirty |= ImGui::SliderFloat("defocus", &param.defocus, 0.0f, 10.0f);
            dirty |= ImGui::SliderInt("num lamdas", &param.num_lambdas, 1, 20);
            dirty |= ImGui::SliderFloat("exposure", &param.exposure, 0.0f, 1.0f);
            dirty |= ImGui::RadioButton("None", &lens_file, 0);
            dirty |= ImGui::RadioButton("50mm-lens", &lens_file, 1);
            dirty |= ImGui::RadioButton("Edmund-Optics-achromat-NT32-921", &lens_file, 2);
            dirty |= ImGui::RadioButton("Edmund-Optics-NT49-291", &lens_file, 3);
            dirty |= ImGui::RadioButton("exp001", &lens_file, 4);
            dirty |= ImGui::RadioButton("exp002", &lens_file, 5);
            dirty |= ImGui::RadioButton("exp003", &lens_file, 6);
            dirty |= ImGui::RadioButton("exp004", &lens_file, 7);
            dirty |= ImGui::RadioButton("experiment1", &lens_file, 8);
            dirty |= ImGui::RadioButton("Tessar-Brendel", &lens_file, 9);
            dirty |= ImGui::RadioButton("xenon", &lens_file, 10);

            std::string lensfile[11] =
            {
                "",
                "50mm-lens.lens",
                "Edmund-Optics-achromat-NT32-921.lens",
                "Edmund-Optics-NT49-291.lens",
                "exp001.lens",
                "exp002.lens",
                "exp003.lens",
                "exp004.lens",
                "experiment1.lens",
                "Tessar-Brendel.lens",
                "xenon.lens"
            };

            param.system_definition_file = GetDirectoryWithPackage() + "\\lens\\" + lensfile[lens_file];

            //float fovy = 2.0f * std::atan(param.sensor_width / (2.0f * param.distance));
            //fovy = Degrees(fovy);
            //track_ball_camera.GetCamera().SetFovy(fovy);
            param.dirty = dirty;
        }
    }

    void UpdateDenoise(InitParam& param)
    {
        if (ImGui::CollapsingHeader("Denoise Param")) 
        {
            ImGui::Checkbox("enable denoise", &param.enable_denoise);
        }
    }

    void UpdateNoise() 
    {
        if (ImGui::CollapsingHeader("Noise Param"))
        {
            if (ImGui::Button("generate noise"))
            {
                CMJSeed seed;
                std::vector<float> tmp_image = {};
                int width = 64;
                int height = 64;
                tmp_image.resize(width * height);
                for (size_t sample_id = 0; sample_id < 100; sample_id++)
                {
                    seed.launch_index = 0;
                    seed.sample_index = (uint32_t)sample_id;
                    seed.depth = 0;
                    seed.offset = 0;

                    float2 value = random_cmj2(seed);
                    value.x *= 64;
                    value.y *= 64;

                    int x = (int)value.x;
                    int y = (int)value.y;
                    tmp_image.at(y * 64 + x) = 1.0f;
                }

                DirectX::Image image = {};
                image.width = width;
                image.height = height;
                image.format = DXGI_FORMAT_R32_FLOAT;
                image.rowPitch = width * 1 * sizeof(float);
                image.slicePitch = height * width * 1 * sizeof(float);
                image.pixels = reinterpret_cast<uint8_t*>(tmp_image.data());
                DirectX::SaveToDDSFile(image, DirectX::DDS_FLAGS_NONE, L"./test.dds");
            }
        }
    }

    void End() 
    {
        ImGui::End();
    }

    void Terminate() 
    {
        
    }
};

ImguiTask::ImguiTask() 
    : m_impl(new ImguiTask::Implement)
{
}

ImguiTask::~ImguiTask() 
{
    
}

bool ImguiTask::Initialzie() 
{
    m_impl->Initialize();
    return true;
}

void ImguiTask::Update(GraphicsManager& manager, ApplicationTimeConfig& time_config)
{
    m_impl->Begin();
    m_impl->UpdateTimeConfig(time_config);
    m_impl->UpdateTrackballCamera(manager.GetTrackballCamera());
    m_impl->UpdateInitParam(manager.GetInitParam());
    m_impl->UpdateDebugParam(manager.GetInitParam());
    m_impl->UpdateSkyParam(manager.GetInitParam());
    m_impl->UpdateLens(manager.GetInitParam().lens_param, manager.GetTrackballCamera());
    m_impl->UpdateDenoise(manager.GetInitParam());
    m_impl->UpdateNoise();
    m_impl->End();
}

void ImguiTask::Terminate() 
{
    m_impl->Terminate();
}
} // namespace slug
#endif MODE_DEBUG