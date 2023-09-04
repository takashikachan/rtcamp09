/**
 * @file    GraphicsManager.cpp
 * @brief   グラフィックス処理の管理クラス
 */

#include "graphic/GraphicManager.hpp"
#include "scene/SampleScene.hpp"
#include "utility/SystemPath.hpp"


namespace slug
{


void InitCameraState(Camera& camera)
{
    /*
    camera.SetPosition(make_float3(-278.000, -274.400, -1251.956));
    camera.SetRegardPosition(make_float3(-278.000, -274.400, -1250.956));
    camera.SetFovy(45.0f);
    */
    camera.SetPosition(make_float3(-19.1514, -13.564, 0.273259));
    camera.SetRegardPosition(make_float3(-18.2688, -13.6746, -6.0186));
    camera.SetFovy(45.0f);


    /*
    camera.SetPosition(make_float3(14.0f, 11.5f, -190.0f));
    camera.SetRegardPosition(make_float3(14.0f, 11.5f, 0.0f));
    camera.SetUpVector(make_float3(0.0f, 1.0f, 0.0f));
    camera.SetFovy(7.959f);
    */
    camera.drity = true;
}

void InitTrackballState(Trackball& trackball) 
{
    InitCameraState(trackball.GetCamera());
    float3 u, v, w;
    u = make_float3(1.0f, 0.0f, 0.0f);
    v = make_float3(0.0f, 1.0f, 0.0f);
    w = make_float3(0.0f, 0.0f, 1.0f);

    trackball.SetReferenceFrame(u,v,w);
    trackball.SetMoveSpeed(10.0f);
    trackball.SetGimbalLock(true);
    trackball.ReinitOrientationFromCamera();
}

GraphicsManager::GraphicsManager() 
{

}

GraphicsManager::~GraphicsManager() 
{

}

bool GraphicsManager::Initialize(const GraphicsManagerParam& param) 
{
    InitTrackballState(m_traclball_camera);
    m_context.CreateContext(param.enable_debug);

    m_init_param.width = param.render_width;
    m_init_param.height = param.render_height;
    m_init_param.output_type = OutputType::Sdr;
    m_init_param.enable_denoise = true;
    m_init_param.sample_per_launch = 1;
    m_init_param.max_depth = 2;
    m_init_param.sky_intensity = 0.01f;
    m_init_param.sun_dir[0] = 0.5f;
    m_init_param.sun_dir[1] = 1.0f;
    m_init_param.sun_dir[2] = -0.6f;
    m_init_param.sun_emission[0] = 1.0f;
    m_init_param.sun_emission[1] = 1.0f;
    m_init_param.sun_emission[2] = 1.0f;

    m_init_param.lens_param.enable_lensflare = true;
    m_init_param.lens_param.anamorphic = 3;
    m_init_param.lens_param.degree = 3;
    m_init_param.lens_param.distance = 26.7f;
    m_init_param.lens_param.sample_mul = 50.0f;
    m_init_param.lens_param.r_entrance = 2.5f;
    m_init_param.lens_param.defocus = 0.0f;
    m_init_param.lens_param.num_lambdas = 8;
    m_init_param.lens_param.exposure = 0.01f;
    m_init_param.lens_param.system_definition_file = GetDirectoryWithPackage() + "\\lens\\Tessar-Brendel.lens";
    m_scene.Initalize(m_context, m_init_param);
   
    m_output_buffer.Create(CudaPixelBufferType::CUDA_DEVICE, param.render_width, param.render_height);
    m_hdr_buffer.Create(CudaPixelBufferType::CUDA_DEVICE, param.render_width, param.render_height);

    return true;
}

void GraphicsManager::Update(uint32_t framecount) 
{

    m_scene.UpdateState(m_context, m_output_buffer, m_traclball_camera.GetCamera(), false, m_init_param,  m_init_param.framecount);

    LaunchArg arg;
    arg.output_buffer = &m_output_buffer;
    arg.hdr_buffer = &m_hdr_buffer;
    m_scene.LaunchSubframe(arg);
}

void GraphicsManager::Terminate() 
{
    m_scene.CleanupState();
    m_context.Cleanup();
}

void GraphicsManager::GetOutputBuffer(ImageBuffer& output)
{
    output.width = m_output_buffer.GetWidth();
    output.height = m_output_buffer.GetHeight();
    output.data = m_output_buffer.GetHostPointer();
    output.pixel_format = BufferImageFormat::Ubyte4;
}

Trackball& GraphicsManager::GetTrackballCamera()
{
    return m_traclball_camera;
}

InitParam& GraphicsManager::GetInitParam()
{
    return m_init_param;
}
} // namespace slug