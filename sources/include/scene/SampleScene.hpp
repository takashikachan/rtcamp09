#pragma once

#include "scene/IScene.hpp"

namespace slug
{
class SampleScene
{
public:
    SampleScene();
    ~SampleScene();
    void Initalize(GraphicsContext& context, const InitParam& param);
    void UpdateState(GraphicsContext& context, SdrPixelBuffer& output_buffer, Camera& camera, bool resize_dirty, InitParam& param, uint32_t framecount);
    void LaunchSubframe(LaunchArg& arg);
    void CleanupState();
private:
    void LoadScene(GraphicsContext& context);
    void SetupObject(GraphicsContext& context);
    void CreateModule(GraphicsContext& context);
    void CreateCudaModule(GraphicsContext& context);
    void CreateProgramGroups(GraphicsContext& context);
    void CreatePipeline(GraphicsContext& context);
    void CreateSBT(GraphicsContext& context);
    void InitLaunchParams(GraphicsContext& context, const InitParam& param);
    void InitDenoiser(GraphicsContext& context, const InitParam& param);
private:
    void UpdateAnimation(int32_t framecount, Camera& camera);
    void UpdateParticle(GraphicsContext& context, uint32_t framecount);
    void UpdateLaunchParam(SdrPixelBuffer& output_buffer, InitParam& param);
    bool UpdateObject(GraphicsContext& context);
    bool UpdateLight(GraphicsContext& context);
    bool UpdateLensSystem(GraphicsContext& context);
    bool UpdateCamera(Camera& camera);
    bool UpdateResize(SdrPixelBuffer& output_buffer, bool resize_dirty);
    bool UpdateSky(InitParam& param);
    void UpdateIAS(GraphicsContext& context, bool is_update);
    void UpdateSubframe(bool is_reset);
    void CopyOutputBuffer(LaunchArg& arg);
    void LaunchCudaKernel(LaunchArg& arg);
    void LaunchOptixKernel(LaunchArg& arg);
    void TonemapPass(int width, int height);
    void CopyBufferPass(int width, int height);
    void LensSystemPass(int width, int height);
    void OptixDenoiserPass();
    void AtrousWaveletPass(int width, int height);
private:
    struct Implement;
    std::unique_ptr<Implement> m_impl = nullptr;
};
} // namespace slug