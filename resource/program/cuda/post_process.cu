#include <cuda/helpers.h>
#include <optix_slug/random.h>
#include <optix_slug/cmj_utility.h>
#include <optix_slug/lens_system.h>

static __forceinline__ __device__ float ACESFilimic(float x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    float v = (x * (a * x + b)) / (x * (c * x + d) + e);
    return clamp(v, 0.0f, 1.0f);
}

extern "C" __global__ void launch_tonemap(float4* input, uchar4* output, int numElements) 
{

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) 
  {
    float gamma = 1.0 / 2.2f;
    float4 in_color = input[i];
    in_color.x = pow(ACESFilimic(in_color.x), gamma);
    in_color.y = pow(ACESFilimic(in_color.y), gamma);
    in_color.z = pow(ACESFilimic(in_color.z), gamma);
    uchar4 out_color = make_color(in_color);
    output[i] = out_color;
  }
}


extern "C" __global__ void launch_copybuffer(float4* input, uchar4* output, int numElements) 
{

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) 
  {
    float4 in_color = input[i];
    uchar4 out_color = make_color(in_color);
    output[i] = out_color;
  }
}

extern "C" __global__ void launch_lensflare(float4* out_color, float4* in_color, int numElements, int width, int height, TruncPolySystem33* systems, int lambda_samples, float* rgb_spctorum, LensParam* lens_param, float* blade_positions)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int x = index % width;
  int y = index / width;
  if (index >= numElements) 
  {
    return;
  }

  float lambda_from = lens_param[0].lambda_from;
  float lambda_to = lens_param[0].lambda_to;
  float sensor_width = lens_param[0].sensor_width;
  float magnification = lens_param[0].magnification;
  float r_pupil = lens_param[0].r_pupil;
  float pixel_size = lens_param[0].pixel_size;
  float anamorphic = lens_param[0].anamorphic;
  float sensor_scaling = lens_param[0].sensor_scaling;
  float sensor_xres = lens_param[0].sensor_xres;
  float sensor_yres = lens_param[0].sensor_yres;
  float sample_mul = lens_param[0].sample_mul;
  float exposure = lens_param[0].exposure;
  int blade_count = lens_param[0].blade_count;

  CMJSeed seed;
  seed.launch_index = index;
  
  float3 tmp_color = make_float3(0.0f, 0.0f, 0.0f);
  for (int ll = 0; ll < lambda_samples; ++ll) 
  {
    seed.sample_index = ll;  
    float lambda = lambda_from + (lambda_to - lambda_from) * (ll / (float)(lambda_samples - 1));
    int system_index = ll * height + y;
    TruncPolySystem33& system = systems[system_index];

    const float x_sensor = (x / (float)width - 0.5f) * sensor_width;
    const float x_world = x_sensor / magnification;
    const float3 rgbin = make_float3(in_color[index]) * exposure;
    float L_in = spectrum_rgb_to_p(lambda, rgbin);
    int num_samples = max(1, (int)(L_in * sample_mul));
    float sample_weight = L_in / num_samples;
    for (int sample = 0; sample < num_samples; ++sample) 
    {
      seed.depth = sample;
      float2 ap = SamplingBladePosition(blade_positions, blade_count, seed, r_pupil);
      float x_ap = ap.x;
      float y_ap = ap.y;

      float in[5], out[4];

      in[0] = x_world + pixel_size * (random_cmj1(seed) - 0.5f);
      in[1] = x_ap / anamorphic;
      in[2] = y_ap;

      evaluatelens(system, in, out);

      out[0] = out[0] * sensor_scaling + sensor_xres / 2;
      out[1] = out[1] * sensor_scaling + sensor_yres / 2;

      float lambert = sqrt(1 - out[2]);
      if (lambert != lambert) lambert = 0;

      int out_x = (int)out[0];
      int out_y = (int)out[1];
      int out_index = out_y * width + out_x;
      if(out_index > 0 && out_index < numElements)
      {
        float r = lambert * sample_weight * rgb_spctorum[0 + ll * 3];
        float g = lambert * sample_weight * rgb_spctorum[1 + ll * 3];
        float b = lambert * sample_weight * rgb_spctorum[2 + ll * 3];

        //float r = L_in * rgb_spctorum[0 + ll * 3];
        //float g = L_in * rgb_spctorum[1 + ll * 3];
        //float b = L_in * rgb_spctorum[2 + ll * 3];

        tmp_color = make_float3(r, g, b);
        out_color[out_index] += make_float4(tmp_color.x, tmp_color.y, tmp_color.z, 1.0f);
      }
    }
  }
}