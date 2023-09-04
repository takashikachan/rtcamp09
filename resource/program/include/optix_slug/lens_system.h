#pragma once

static __device__ float spectrum_s_white[] =
{
  1.0000, 1.0000, 0.9999, 0.9993, 0.9992, 0.9998, 1.0000, 1.0000, 1.0000, 1.0000
};
static __device__ float spectrum_s_cyan[] =
{
  0.9710, 0.9426, 1.0007, 1.0007, 1.0007, 1.0007, 0.1564, 0.0000, 0.0000, 0.0000
};
static __device__ float spectrum_s_magenta[] =
{
  1.0000, 1.0000, 0.9685, 0.2229, 0.0000, 0.0458, 0.8369, 1.0000, 1.0000, 0.9959 
};
static __device__ float spectrum_s_yellow[] =
{
  0.0001, 0.0000, 0.1088, 0.6651, 1.0000, 1.0000, 0.9996, 0.9586, 0.9685, 0.9840 
};
static __device__ float spectrum_s_red[] =
{
  0.1012, 0.0515, 0.0000, 0.0000, 0.0000, 0.0000, 0.8325, 1.0149, 1.0149, 1.0149 
};
static __device__ float spectrum_s_green[] =
{
  0.0000, 0.0000, 0.0273, 0.7937, 1.0000, 0.9418, 0.1719, 0.0000, 0.0000, 0.0025 
};
static __device__ float spectrum_s_blue[] =
{
  1.0000, 1.0000, 0.8916, 0.3323, 0.0000, 0.0000, 0.0003, 0.0369, 0.0483, 0.0496
};

__host__ __device__ __inline__ float spectrum_rgb_to_p(float lambda, const float3 rgb)
{
    // smits-like smooth metamer construction, basis function match cie rgb backwards.
    float p = 0.0f;
    float red = rgb.x;
    float green = rgb.y;
    float blue = rgb.z;

    float cyan = 0;
    float yellow = 0;
    float magenta = 0;

    const float white = min(red, min(green, blue));
    red -= white; 
    green -= white; 
    blue -= white;

    const int bin = (int)(10.0f*(lambda - 380.0f)/(720.0 - 380.0));

    float ww = spectrum_s_white[bin];
    p += white * ww;
    
    if(green > 0 && blue > 0)
    {
        cyan = min(green, blue);
        green -= cyan; 
        blue -= cyan;
    }
    else if(red > 0 && blue > 0)
    {
        magenta = min(red, blue);
        red -= magenta; 
        blue -= magenta;
    }
    else if(red > 0 && green > 0)
    {
        yellow = min(red, green);
        red -= yellow; 
        green -= yellow;
    }

    float cw = spectrum_s_cyan[bin];
    float mw = spectrum_s_magenta[bin];
    float yw = spectrum_s_yellow[bin];
    p += cw*cyan;
    p += mw*magenta;
    p += yw*yellow;

    float rw = spectrum_s_red[bin];
    float gw = spectrum_s_green[bin];
    float bw = spectrum_s_blue[bin];
    p += red * rw;
    p += green * gw;
    p += blue * bw;

    return p;
}

struct Camera
{
  float3 U;
  float3 V;
  float3 W;
};

struct PolyTerm3
{
  float coefficient;
  int exponents[3];
};

struct TruncPoly3
{    
  int trunc_degree;
  PolyTerm3 terms[30];
  int term_count;
  bool consolidated;
};

struct TruncPolySystem33
{
    TruncPoly3 equations[3]; 
    int trunc_degree;
};

struct LensParam
{
  float lambda_from;
  float lambda_to;
  float sensor_width;
  float magnification;
  float r_pupil;
  float pixel_size;
  float anamorphic;
  float sensor_scaling;
  float sensor_xres;
  float sensor_yres;
  float sample_mul;
  float exposure;
  float blade_count;
};

static __forceinline__ __device__  void evaluatelens(TruncPolySystem33& system, float* x0, float* x1)
{
  for (int i = 0; i < 3; ++i) 
  {
    float result = 0;
    if(system.equations[i].term_count > 0)
    {
      for (int j = (int)system.equations[i].term_count - 1; j >= 0; --j) 
      {
        PolyTerm3& term = system.equations[i].terms[j];
        float term_value = term.coefficient;
        for (int k = 0; k < 3; ++k)
        {
          term_value *= pow(x0[k], term.exponents[k]);
        }
        
        result += term_value;
      }
    }
    x1[i] = result;
  }
}

static __forceinline__ __device__ float2 SamplingBladePosition(float* blade_positions, int blade_count, CMJSeed& seed, float r_pupil)
{

  float x_ap = 0.0f;
  float y_ap = 0.0f;
  if (blade_count == 0) 
  {
      do 
      {
          float2 rand_value = random_cmj2(seed);
          x_ap = (rand_value.x - 0.5f) * 2.0f * r_pupil;
          y_ap = (rand_value.y - 0.5f) * 2.0f * r_pupil;
      } while (x_ap * x_ap + y_ap * y_ap > r_pupil * r_pupil);
  }
  else 
  {
      bool inside;
      do 
      {
          inside = true;
          float2 rand_value = random_cmj2(seed);
          x_ap = (rand_value.x - 0.5f) * 2.0f * r_pupil;
          y_ap = (rand_value.y - 0.5f) * 2.0f * r_pupil;

          for (int b = 0; b < blade_count; ++b) 
          {
              float bx = blade_positions[((b + 1) % blade_count) * 2] - blade_positions[b * 2];
              float by = blade_positions[((b + 1) % blade_count) * 2 + 1] - blade_positions[b * 2 + 1];

              float px = x_ap - blade_positions[b * 2];
              float py = y_ap - blade_positions[b * 2 + 1];
              float det = (px * by) - (py * bx);
              if (det > 0) 
              {
                  inside = false;
                  break;
              }
          }
      } while (!inside);
  }
  return make_float2(x_ap, y_ap);
}