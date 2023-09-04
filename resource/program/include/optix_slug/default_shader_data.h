#pragma once

/**
 * @brief レイの種類
*/
enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

struct ArHosekSkyModelConfiguration
{
    float value[9];
};

struct ArHosekSkyModelState
{
    ArHosekSkyModelConfiguration configs[11];
    float radiances[11];
    float turbidity;
    float solar_radius;
    float emission_correction_factor_sky[11];
    float emission_correction_factor_sun[11];
    float albedo;
    float elevation;

    float3 betaR0; // mie
    float3 betaM0; // rayleigh
};

struct Material
{
    float base_color[3];
    float emission[3];
    float ior;
    float relative_ior;
    float specular_tint;
    float specular_trans;
    float sheen;
    float sheen_tint;
    float roughness;
    float metallic;
    float clearcoat;
    float clearcoat_gloss;
    float subsurface ;
    float anisotropic;
    float debug_diffuse;
    float debug_specular;
    cudaTextureObject_t albedo;
    cudaTextureObject_t bump;
};

/**
 * @brief カメラデータ
*/
struct CameraParam
{
    float3       eye;   //!< 視点
    float3       U;     //!< 横方向ベクトル
    float3       V;     //!< 縦方向ベクトル
    float3       W;     //!< 正面方向ベクトル
};

/**
 * @brief デバッグモード
*/
enum DebugMode
{
    DebugMode_None = 0,
    DebugMode_RGB = 1,      //!< RGBレンダリングモード
    DebugMode_Spectrum = 2, //!< スペクトル確認モード
    DebugMode_Color = 3,    //!< 色確認モード
    DebugMode_Noise = 4,    //!< ノイズ確認モード
    DebugMode_Texture = 5   //!< テクスチャ確認モード
};

/**
 * @brief デバッグデータ
*/
struct DebugParam
{
    int debug_mode;                 //!< デバッグモード
    bool russian_roulette;
    float3 color;                   //!< 色
    cudaTextureObject_t texture;    //!< 表示用テクスチャ
};

struct DirectLight
{
    float3 dir;
    float3 emission;
    ArHosekSkyModelState sky_state;
    float sky_intensity;
};

enum LightType
{
    SphereLight = 0,
};

struct AnyLight
{
    LightType type;
    float3 position;
    float radius;
    float3 emission;
};

/**
 * @brief 開始パラメータ
*/
struct LaunchParam
{
    unsigned int width;                 //!< 横サイズ
    unsigned int height;                //!< 縦サイズ

    unsigned int subframe_index;        //!< フレームインデックス
    unsigned int samples_per_launch;    //!< 1フレームにサンプリングする回数
    unsigned int spectrum_samples;      //!< スペクトルのサンプリング数
    unsigned int max_depth;             //!< 最大バウンス
    float min_dist;                     //!< レイの最小距離
    float max_dist;                     //!< レイの最大距離

    float4* accum_buffer;               //!< 履歴バッファ
    float4* albedo_buffer;              //!< アルベドバッファ
    float4* normal_buffer;              //!< 法線バッファ
    float4* position_buffer;            //!< 座標バッファ
    float4* frame_buffer;               //!< 最終出力バッファ

    CameraParam camera;                 //!< カメラ情報
    DebugParam debug;                   //!< デバッグ情報
    DirectLight direct_light;           //!< 直接光情報
    OptixTraversableHandle handle;      //!< ルートノード
    Material* materials;                //!< マテリアルデータ

    AnyLight* lights;                   //!< ライトデータ
    unsigned int light_count;           //!< ライトの数
};

/**
 * @brief RayGenのSBTデータ
*/
struct RayGenData
{
};

/**
 * @brief MissDataのSBTデータ
*/
struct MissData
{
    float4 bg_color;                //!< 背景色
    cudaTextureObject_t env_tex;    //!< 環境テクスチャ
};

/**
 * @brief HitGroupプログラムのSBTデータ
*/
struct HitGroupData
{
    float4 local0;                      //!< ワールド座標行列の1列目
    float4 local1;                      //!< ワールド座標行列の2列目
    float4 local2;                      //!< ワールド座標行列の3列目
    float4 normal0;                     //!< ワールド法線行列の1列目
    float4 normal1;                     //!< ワールド法線行列の2列目
    float4 normal2;                     //!< ワールド法線行列の3列目
    float sbt_index;                    //!< sbt_index
    float3* vertices;                   //!< 座標データ
    float3* normals;                    //!< 法線データ
    float2* texcoords;                  //!< UVデータ
    unsigned int* material_ids;          //!< マテリアルidデータ
    unsigned int* indices;              //!< インデックスデータ
};
