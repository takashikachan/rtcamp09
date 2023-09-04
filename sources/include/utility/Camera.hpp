/**
 * @file    Camera.hpp
 * @brief   描画で使用するカメラクラスの定義ファイル
 */
#pragma once

#include "utility/Define.hpp"
#include "math/Vector.hpp"

namespace slug
{

/**
 * @brief 描画で使用するカメラクラス
*/
class Camera final
{
public:
    /**
     * @brief コンストラクタ
    */
    Camera()
        :m_position(make_float3(1.0f, 1.0f, 1.0f))
        ,m_regard_position(make_float3(0.0f, 0.0f, 0.0f))
        ,m_up_vector(make_float3(0.0f, 1.0f, 0.0f))
        ,m_fovy(35.0f)
        ,m_aspect(1.0f)
    {};

    /**
     * @brief コンストラクタ
    */
    Camera(const float3& position, const float3& regard_position, const float3& up_vector, float fovy, float aspect)
        :m_position(position)
        , m_regard_position(regard_position)
        , m_up_vector(up_vector)
        , m_fovy(fovy)
        , m_aspect(aspect)
    {};

    /**
     * @brief デストラクタ
    */
    ~Camera()
    {}

    /**
     * @brief 方向を取得
    */
    float3 GetDirection() const
    {
        return Normalize(m_regard_position - m_position);
    }

    /**
     * @brief カメラの位置を取得
    */
    float3 GetPosition() const
    {
        return m_position;
    }

    /**
     * @brief カメラの注視点を取得
    */
    float3 GetRegardPosition() const
    {
        return m_regard_position;
    }

    /**
     * @brief カメラの上方向のベクトルを取得
    */
    float3 GetUpVector() const
    {
        return m_up_vector;
    }

    /**
     * @brief カメラの画角を取得
    */
    float GetFovy() const
    {
        return m_fovy;
    }

    /**
     * @brief aspect比を取得
    */
    float GetAspect() const
    {
        return m_aspect;
    }

    /**
     * @brief 方向を設定
    */
    void SetDirection(const float3& direction)
    {
        m_regard_position = m_position + Length(m_regard_position - m_position) * direction;
    }

    /**
     * @brief カメラの位置を設定
    */
    void SetPosition(const float3& position)
    {
        m_position = position;
    }

    /**
     * @brief カメラの注視点を設定
    */
    void SetRegardPosition(const float3& regard_position)
    {
        m_regard_position = regard_position;
    }

    /**
     * @brief カメラの上方向のベクトルを設定
    */
    void SetUpVector(const float3& up_vector)
    {
        m_up_vector = up_vector;
    }

    /**
     * @brief カメラの画角を設定
    */
    void SetFovy(float fovy)
    {
        m_fovy = fovy;
    }

    /**
     * @brief カメラのアスペクト比を設定
    */
    void SetAspect(float aspect)
    {
        m_aspect = aspect;
    }

    /**
     * @brief カメラ空間を表す3つのベクトル(3x3行列)を取得
    */
    void GetCameraSpace(float3& u, float3& v, float3& w) const;
public:
    bool drity = false;
private:
    float3 m_position;          //!< 位置
    float3 m_regard_position;   //!< 注視点
    float3 m_up_vector;         //!< 上方向のベクトル
    float m_fovy;               //!< 画角(Degrees)
    float m_aspect;             //!< アスペクト比
};

class Trackball
{
public:
    enum ViewMode
    {
        EyeFixed,
        LookAtFixed
    };

public:
    bool CalcWheelEvent(int32_t dir);
    void CalcZoom(int32_t direction);
    void StartTracking(int32_t x, int32_t y);
    void UpdateTracking(int32_t x, int32_t y, int32_t canvas_width, int32_t canvas_height);
    void ReinitOrientationFromCamera();
    void SetReferenceFrame(const float3& u, const float3& v, const float3& w);
public:
    float MoveSpeed() const 
    { 
        return m_move_speed; 
    }

    void SetMoveSpeed(const float& val) 
    { 
        m_move_speed = val; 
    }

    Camera& GetCamera() 
    { 
        return m_camera;
    }

    bool GetGimbalLock() const 
    { 
        return m_gimbal_lock; 
    }

    void SetGimbalLock(bool val) 
    { 
        m_gimbal_lock = val; 
    }

    ViewMode GetViewMode() const 
    { 
        return m_view_mode; 
    }

    void SetViewMode(ViewMode val) 
    { 
        m_view_mode = val; 
    }

    void MoveForward(float speed);
    void MoveBackward(float speed);
    void MoveLeft(float speed);
    void MoveRight(float speed);
    void MoveUp(float speed);
    void MoveDown(float speed);
private:
    void UpdateCamera();
    void RollLeft(float speed);
    void RollRight(float speed);
private:
    bool         m_gimbal_lock = false;
    ViewMode     m_view_mode = LookAtFixed;
    float        m_camera_eye_lookat_distance = 0.0f;
    float        m_zoom_multiplier = 1.1f;
    float        m_move_speed = 1.0f;
    float        m_roll_speed = 0.5f;

    float        m_latitude = 0.0f;
    float        m_longitude = 0.0f;

    int          m_prev_pos_x = 0;
    int          m_prev_pos_y = 0;
    bool         m_perform_tracking = false;

    float3       m_u = { 0.0f, 0.0f, 0.0f };
    float3       m_v = { 0.0f, 0.0f, 0.0f };
    float3       m_w = { 0.0f, 0.0f, 0.0f };

    Camera m_camera = {};
};
} // namespace slug