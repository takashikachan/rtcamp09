/**
 * @file    Camera.cpp
 * @brief   描画で使用するカメラクラスのソースファイル
 */

#include "utility/Camera.hpp"

namespace slug
{
void Camera::GetCameraSpace(float3& u, float3& v, float3& w) const
{
    w = m_regard_position - m_position;
    float wlen = Length(w);
    u = Normalize(Cross(w, m_up_vector));
    v = Normalize(Cross(u, w));

    float vlen = wlen * tanf(0.5f * m_fovy * M_PIf / 180.0f);
    v = v * vlen;
    float ulen = vlen * m_aspect;
    u = u * ulen;
}

void Trackball::StartTracking(int32_t x, int32_t y)
{
    m_prev_pos_x = x;
    m_prev_pos_y = y;
    m_perform_tracking = true;
}

void Trackball::UpdateTracking(int32_t x, int32_t y, int32_t canvas_width, int32_t canvas_height)
{
    if (!m_perform_tracking)
    {
        StartTracking(x, y);
        return;
    }

    int32_t delta_x = x - m_prev_pos_x;
    int32_t delta_y = y - m_prev_pos_y;

    m_prev_pos_x = x;
    m_prev_pos_y = y;
    m_latitude = Radians(TMin(89.0f, TMax(-89.0f, Degrees(m_latitude) + 0.5f * delta_y)));
    m_longitude = Radians(fmod(Degrees(m_longitude) - 0.5f * delta_x, 360.0f));

    UpdateCamera();

    if (!m_gimbal_lock) 
    {
        ReinitOrientationFromCamera();
        m_camera.SetUpVector(m_w);
    }
}

void Trackball::ReinitOrientationFromCamera()
{
    m_camera.GetCameraSpace(m_u, m_v, m_w);
    m_u = Normalize(m_u);
    m_v = Normalize(m_v);
    m_w = Normalize(-m_w);
    std::swap(m_v, m_w);
    m_latitude = 0.0f;
    m_longitude = 0.0f;
    m_camera_eye_lookat_distance = Length(m_camera.GetRegardPosition() - m_camera.GetPosition());
}

void Trackball::SetReferenceFrame(const float3& u, const float3& v, const float3& w)
{
    m_u = u;
    m_v = v;
    m_w = w;
    float3 dirWS = -Normalize(m_camera.GetRegardPosition() - m_camera.GetPosition());
    float3 dirLocal;
    dirLocal.x = Dot(dirWS, u);
    dirLocal.y = Dot(dirWS, v);
    dirLocal.z = Dot(dirWS, w);
    m_longitude = atan2(dirLocal.x, dirLocal.y);
    m_latitude = asin(dirLocal.z);
}

void Trackball::CalcZoom(int32_t direction)
{
    float zoom = (direction > 0) ? 1 / m_zoom_multiplier : m_zoom_multiplier;
    m_camera_eye_lookat_distance *= zoom;
    float3 lookat = m_camera.GetRegardPosition();
    float3 camera_pos = m_camera.GetPosition();
    m_camera.SetPosition(lookat + (camera_pos - lookat) * zoom);
}

bool Trackball::CalcWheelEvent(int32_t dir)
{
    CalcZoom(dir);
    return true;
}

void Trackball::MoveForward(float speed)
{
    float3 dirWS = Normalize(m_camera.GetRegardPosition() - m_camera.GetPosition());
    m_camera.SetPosition(m_camera.GetPosition() + dirWS * speed);
    m_camera.SetRegardPosition(m_camera.GetRegardPosition() + dirWS * speed);
}
void Trackball::MoveBackward(float speed)
{
    float3 dirWS = Normalize(m_camera.GetRegardPosition() - m_camera.GetPosition());
    m_camera.SetPosition(m_camera.GetPosition() - dirWS * speed);
    m_camera.SetRegardPosition(m_camera.GetRegardPosition() - dirWS * speed);
}

void Trackball::MoveLeft(float speed)
{
    float3 u, v, w;
    m_camera.GetCameraSpace(u, v, w);
    u = Normalize(u);

    m_camera.SetPosition(m_camera.GetPosition() - u * speed);
    m_camera.SetRegardPosition(m_camera.GetRegardPosition() - u * speed);
}

void Trackball::MoveRight(float speed)
{
    float3 u, v, w;
    m_camera.GetCameraSpace(u, v, w);
    u = Normalize(u);

    m_camera.SetPosition(m_camera.GetPosition() + u * speed);
    m_camera.SetRegardPosition(m_camera.GetRegardPosition() + u * speed);
}

void Trackball::MoveUp(float speed)
{
    float3 u, v, w;
    m_camera.GetCameraSpace(u, v, w);
    v = Normalize(v);

    m_camera.SetPosition(m_camera.GetPosition() + v * speed);
    m_camera.SetRegardPosition(m_camera.GetRegardPosition() + v * speed);
}

void Trackball::MoveDown(float speed)
{
    float3 u, v, w;
    m_camera.GetCameraSpace(u, v, w);
    v = Normalize(v);

    m_camera.SetPosition(m_camera.GetPosition() - v * speed);
    m_camera.SetRegardPosition(m_camera.GetRegardPosition() - v * speed);
}

void Trackball::RollLeft(float speed)
{
    float3 u, v, w;
    m_camera.GetCameraSpace(u, v, w);
    u = Normalize(u);
    v = Normalize(v);

    m_camera.SetUpVector(u * cos(Radians(90.0f + speed)) + v * sin(Radians(90.0f + speed)));
}

void Trackball::RollRight(float speed)
{
    float3 u, v, w;
    m_camera.GetCameraSpace(u, v, w);
    u = Normalize(u);
    v = Normalize(v);

    m_camera.SetUpVector(u * cos(Radians(90.0f - speed)) + v * sin(Radians(90.0f - speed)));
}

void Trackball::UpdateCamera()
{
    float3 local_dir;
    local_dir.x = cos(m_latitude) * sin(m_longitude);
    local_dir.y = cos(m_latitude) * cos(m_longitude);
    local_dir.z = sin(m_latitude);

    float3 dir_ws = m_u * local_dir.x + m_v * local_dir.y + m_w * local_dir.z;

    if (m_view_mode == EyeFixed)
    {
        float3 position = m_camera.GetPosition();
        m_camera.SetRegardPosition(position - dir_ws * m_camera_eye_lookat_distance);
    }
    else
    {
        float3 lookat = m_camera.GetRegardPosition();
        m_camera.SetPosition(lookat + dir_ws * m_camera_eye_lookat_distance);
    }
}
} // namespace slug
