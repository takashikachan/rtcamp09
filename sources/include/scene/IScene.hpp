#pragma once

#include "graphic/GraphicsContext.hpp"
#include "utility/LensSystem.hpp"

namespace slug
{
struct LaunchArg
{
    SdrPixelBuffer* output_buffer;
    HdrPixelBuffer* hdr_buffer;
};

enum class OutputType : uint8_t
{
    Albedo,
    Normal,
    Position,
    Sdr,
    Hdr
};

struct InitParam
{
    uint32_t width = 0;
    uint32_t height = 0;
    int sample_per_launch = 30;
    int max_depth = 5;
    int debug_mode = 0;
    bool russian_roulette = true;
    float debug_color[3] = { 0.0f, 0.0f, 0.0f };

    float sun_dir[3] = { -0.5f, 0.5f, -1.0f };
    float sun_emission[3] = { 1.0f, 1.0f, 1.0f };
    float atmospheric_turbidity = 2.0f;
    float ground_albedo = 1.0f;
    float sky_intensity = 0.01f;
    bool sky_dirty = true;

    OutputType output_type = OutputType::Albedo;
    int framecount = 0;
    bool enable_denoise = true;
    LensSystemParam lens_param = {};
};

class IScene {
public:
    IScene() = default;
    ~IScene() = default;
    virtual void Initalize(GraphicsContext& context, const InitParam& param) = 0;
    virtual void UpdateState(SdrPixelBuffer& output_buffer, Camera& camera, bool resize_dirty, InitParam& param) = 0;
    virtual void LaunchSubframe(LaunchArg& arg) = 0;
    virtual void CleanupState() = 0;
};
}