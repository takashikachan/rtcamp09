#ifndef _TEST_0_HEADER_
#define _TEST_0_HEADER_

struct Params
{
    uchar4* image;
    unsigned int           image_width;
    unsigned int           image_height;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};


struct RayGenData
{
    // No data needed
};
#endif