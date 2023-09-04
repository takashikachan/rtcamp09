#include <TruncPoly/TruncPolySystem.hh>

#include <OpticalElements/OpticalMaterial.hh>
#include <OpticalElements/Spherical5.hh>
#include <OpticalElements/Cylindrical5.hh>
#include <OpticalElements/Propagation5.hh>
#include <OpticalElements/TwoPlane5.hh>

#include <OpticalElements/FindFocus.hh>

#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <ostream>
#define cimg_display 0

#include <omp.h> 

#include <spectrum.h>
#include <fstream>
#include "utility/LensSystem.hpp"

namespace slug
{
class CImage
{
public:
    CImage(int32_t w, int32_t h, int32_t d, int32_t c, float value = 0) 
        : m_width(w)
        , m_height(h)
        , m_depth(d)
        , m_channel(c)
    {
        m_count = m_width * m_height * m_depth * m_channel;
        m_data = new float[m_count];
        m_byte_size = m_count * sizeof(float);
        memset(m_data, 0, m_byte_size);
    }

    ~CImage() 
    {
        if (m_data) 
        {
            delete[] m_data;
        }
    }

    void CorrectPosition(int32_t& x, int32_t& y, int32_t& z, int32_t& c)
    {
        x = (x >= 0) ? x : 0;
        x = (x < m_width) ? x : m_width - 1;

        y = (y >= 0) ? y : 0;
        y = (y < m_height) ? y : m_height - 1;

        z = (z >= 0) ? z : 0;
        z = (z < m_depth) ? z : m_depth - 1;

        c = (c >= 0) ? c : 0;
        c = (c < m_channel) ? c : m_channel - 1;
    }

    bool CheckPosition(int32_t x, int32_t y, int32_t z, int32_t c) const
    {
        bool check_data = true;
        check_data &= (x >= 0) && (x < m_width);
        check_data &= (y >= 0) && (y < m_height);
        check_data &= (z >= 0) && (z < m_depth);
        check_data &= (c >= 0) && (c < m_channel);
        return check_data;
    }

    int32_t CalcIndex(int32_t x, int32_t y, int32_t z, int32_t c) const
    {
        int32_t index = z * m_height * m_width * m_channel + y * m_channel * m_width + x * m_channel + c;
        if (index >= m_count)
        {
            return 0;
        }
        return index;
    }

    float& atXY(int32_t x, int32_t y, int32_t z, int32_t c) 
    {
        CorrectPosition(x, y, z, c);
        int32_t index = CalcIndex(x, y, z, c);
        return m_data[index];
    }

    float atXY(int32_t x, int32_t y, int32_t z, int32_t c, float out_value) const
    {
        if (!CheckPosition(x, y, z, c))
        {
            return out_value;
        }

        int32_t index = CalcIndex(x, y, z, c);
        return m_data[index];
    }

    float linear_atXY(const float fx, const float fy, const int z, const int c, const float out_value) const {
        const int x = (int)fx - (fx >= 0 ? 0 : 1);
        const int nx = x + 1;
        
        const int y = (int)fy - (fy >= 0 ? 0 : 1);
        const int ny = y + 1;

        const float dx = fx - x;
        const float dy = fy - y;

        const float Icc = atXY(x, y, z, c, out_value);
        const float Inc = atXY(nx, y, z, c, out_value);
        const float Icn = atXY(x, ny, z, c, out_value);
        const float Inn = atXY(nx, ny, z, c, out_value);

        return Icc + dx * (Inc - Icc + dy * (Icc + Inn - Icn - Inc)) + dy * (Icn - Icc);
    }

    CImage& set_linear_atXY(const float& value, const float fx, const float fy = 0, const int z = 0, const int c = 0, const bool is_added = false) 
    {
        const int x = (int)fx - (fx >= 0 ? 0 : 1), nx = x + 1;
        const int y = (int)fy - (fy >= 0 ? 0 : 1), ny = y + 1;
        const float dx = fx - x;
        const float dy = fy - y;
        if (z >= 0 && z < depth() && c >= 0 && c < channel()) 
        {
            if (y >= 0 && y < height()) 
            {
                if (x >= 0 && x < width()) 
                {
                    const float w1 = (1 - dx) * (1 - dy);
                    const float w2 = is_added ? 1 : (1 - w1);
                    atXY(x, y, z, c) = (float)(w1 * value + w2 * atXY(x, y, z, c));
                }
                if (nx >= 0 && nx < width()) {
                    const float w1 = dx * (1 - dy);
                    const float w2 = is_added ? 1 : (1 - w1);
                    atXY(nx, y, z, c) = (float)(w1 * value + w2 * atXY(nx, y, z, c));
                }
            }
            if (ny >= 0 && ny < height()) 
            {
                if (x >= 0 && x < width()) 
                {
                    const float w1 = (1 - dx) * dy;
                    const float w2 = is_added ? 1 : (1 - w1);
                    atXY(x, ny, z, c) = (float)(w1 * value + w2 * atXY(x, ny, z, c));
                }

                if (nx >= 0 && nx < width()) 
                {
                    const float w1 = dx * dy;
                    const float w2 = is_added ? 1 : (1 - w1);
                    atXY(nx, ny, z, c) = (float)(w1 * value + w2 * atXY(nx, ny, z, c));
                }
            }
        }
        return *this;
    }

    int32_t width() 
    {
        return m_width;
    }

    int32_t height()
    {
        return m_height;
    }

    int32_t depth() 
    {
        return m_depth;
    }

    int32_t channel() 
    {
        return m_channel;
    }

    float* data() 
    {
        return m_data;
    }
private:
    int32_t m_width = 0;
    int32_t m_height = 0;
    int32_t m_depth = 0;
    int32_t m_channel = 0;
    int32_t m_count = 0;
    int32_t m_byte_size = 0;
    float* m_data;
};
Transform4f get_system(float lambda, int degree) {
    // Let's simulate Edmund Optics achromat #NT32-921:
    /* Clear Aperture CA (mm) 	39.00
    Eff. Focal Length EFL (mm) 	120.00
    Back Focal Length BFL (mm) 	111.00
    Center Thickness CT 1 (mm) 	9.60
    Center Thickness CT 2 (mm) 	4.20
    Radius R1 (mm) 	65.22
    Radius R2 (mm) 	-62.03
    Radius R3 (mm) 	-1240.67
    Substrate 	N-SSK8/N-SF10
    */

    OpticalMaterial glass1("N-SSK8", true);
    OpticalMaterial glass2("N-SF10", true);

    // Also try: const float d0 = 5000; // Scene is 5m away
    const float d0 = 5000000; // Scene is 5km away
    const float R1 = 65.22;
    const float d1 = 9.60;
    const float R2 = -62.03;
    const float d2 = 4.20;
    const float R3 = -1240.67;

    return two_plane_5(d0, degree)
        >> refract_spherical_5(R1, 1.f, glass1.get_index(lambda), degree)
        >> propagate_5(d1, degree)
        >> refract_spherical_5(R2, glass1.get_index(lambda), glass2.get_index(lambda), degree)
        >> propagate_5(d2, degree)
        >> refract_spherical_5(R3, glass2.get_index(lambda), 1.f, degree);
}

Transform4f get_system_from_file(const char* filename, float lambda, int degree, float distance) {
    std::ifstream infile(filename);
    std::string line;

    Transform4f system;
    while (std::getline(infile, line)) {
        std::istringstream ls(line);
        std::string op;
        ls >> op;

        if (op == "two_plane") {
            system = two_plane_5(distance, degree);
            //cout << "two_plane" << " " << d << endl;
        }
        else if (op == "cylindrical_x") {
            float radius;
            std::string glassName1;
            std::string glassName2;
            ls >> radius;
            ls >> glassName1;
            ls >> glassName2;
            float n1 = 1.0f;
            float n2 = 1.0f;
            if (glassName1[0] >= '0' && glassName1[0] <= '9') {
                n1 = (float)atof(glassName1.c_str());

            }
            else {
                OpticalMaterial glass1(glassName1.c_str());
                n1 = glass1.get_index(lambda);
            }

            if (glassName2[0] >= '0' && glassName2[0] <= '9') {
                n2 = (float)atof(glassName2.c_str());
            }
            else {
                OpticalMaterial glass2(glassName2.c_str());
                n2 = glass2.get_index(lambda);
            }

            system = system >> refract_cylindrical_x_5(radius, n1, n2);
        }
        else if (op == "cylindrical_y") {
            float radius;
            std::string glassName1;
            std::string glassName2;
            ls >> radius;
            ls >> glassName1;
            ls >> glassName2;
            float n1 = 1.0f;
            float n2 = 1.0f;
            if (glassName1[0] >= '0' && glassName1[0] <= '9') {
                n1 = (float)atof(glassName1.c_str());

            }
            else {
                OpticalMaterial glass1(glassName1.c_str());
                n1 = glass1.get_index(lambda);
            }

            if (glassName2[0] >= '0' && glassName2[0] <= '9') {
                n2 = (float)atof(glassName2.c_str());
            }
            else {
                OpticalMaterial glass2(glassName2.c_str());
                n2 = glass2.get_index(lambda);
            }

            system = system >> refract_cylindrical_y_5(radius, n1, n2);
        }
        else if (op == "reflect_spherical") {
            float radius;
            ls >> radius;
            system = system >> reflect_spherical_5(radius, degree);
        }
        else if (op == "refract_spherical") {
            float radius;
            std::string glassName1;
            std::string glassName2;
            ls >> radius;
            ls >> glassName1;
            ls >> glassName2;

            float n1 = 1.0f;
            float n2 = 1.0f;

            if (glassName1[0] >= '0' && glassName1[0] <= '9') {
                n1 = (float)atof(glassName1.c_str());

            }
            else {
                OpticalMaterial glass1(glassName1.c_str());
                n1 = glass1.get_index(lambda);
            }

            if (glassName2[0] >= '0' && glassName2[0] <= '9') {
                n2 = (float)atof(glassName2.c_str());
            }
            else {
                OpticalMaterial glass2(glassName2.c_str());
                n2 = glass2.get_index(lambda);
            }

            system = system >> refract_spherical_5(radius, n1, n2, degree);
            //system = system >> refract_cylindrical_x_5(radius, n1, n2, degree);
            //cout << "refract_spherical" << " " << radius << " " << n1 << " " << n2 << endl;

        }
        else if (op == "propagate") {
            float d;
            ls >> d;
            system = system >> propagate_5(d, degree);
            //cout << "propagate" << " " << d << endl;

        }
        else {
            //std::cout << "invalid op: " << op << std::endl;
        }
    }
    return system;
}

struct LensSystem::Impl 
{
    std::unique_ptr<CImage> image_in = nullptr;
    std::unique_ptr<CImage> image_out = nullptr;
    float sensor_width = 0;
    float r_pupil = 0;
    float pixel_size = 0;
    float magnification = 0;
    float sensor_scaling = 0;
    int sensor_xres = 0;
    int sensor_yres = 0;
    std::vector<float> blade_positions = {};
    std::vector<float> rgb = {};
    std::vector<System33f> lambda_system;
};

LensSystem::LensSystem() 
    :m_impl(new Impl)
{
}

LensSystem::~LensSystem() 
{
}

void LensSystem::SetupParam(LensSystemParam& param)
{
    m_param = param;

    std::string system_definition_file = m_param.system_definition_file.c_str();
    int degree = m_param.degree;
    float distance = m_param.distance;

    Transform4f system = {};
    if (system_definition_file.empty())
    {
        system = get_system(550, degree);
    }
    else
    {
        system = get_system_from_file(system_definition_file.c_str(), 550, degree, distance);
    }

    float d3 = find_focus_X(system);
    m_impl->magnification = get_magnification_X(system >> propagate_5(d3));

    m_impl->image_in.reset();
    m_impl->image_in = std::make_unique<CImage>(m_param.width, m_param.height, 1, 4);

    int width = m_impl->image_in->width();
    int height = m_impl->image_in->height();

    m_impl->sensor_width = m_param.sensor_width;
    m_impl->sensor_xres = width;
    m_impl->sensor_yres = height;
    m_impl->sensor_scaling = m_impl->sensor_xres / m_impl->sensor_width;

    Transform4f prop = propagate_5(d3 + m_param.defocus, degree);
    system = system >> prop;

    CImage img_out(m_impl->sensor_xres, m_impl->sensor_yres, 1, 4, 0);
    m_impl->image_out.reset();
    m_impl->image_out = std::make_unique<CImage>(m_impl->sensor_xres, m_impl->sensor_yres, 1, 3);

    m_impl->r_pupil = m_param.r_entrance;
    
    m_impl->rgb.resize(3 * param.num_lambdas);
#pragma omp parallel for
    for (int ll = 0; ll < param.num_lambdas; ++ll)
    {
        float lambda = param.lambda_from + (param.lambda_to - param.lambda_from) * (ll / (float)(param.num_lambdas - 1));
        if (param.num_lambdas == 1)
        {
            lambda = 550;
        }
        spectrum_p_to_rgb(lambda, 1, m_impl->rgb.data() + 3 * ll);
    }


    Transform4d system_spectral_center =
        (!system_definition_file.empty() ? get_system_from_file(system_definition_file.c_str(), 500, degree, distance) : get_system(500, degree))
        >> prop;
    Transform4d system_spectral_right =
        (!system_definition_file.empty() ? get_system_from_file(system_definition_file.c_str(), 600, degree, distance) : get_system(600,
            degree))
        >> prop;

    System54f system_spectral = system_spectral_center.lerp_with(system_spectral_right, 550, 600);

    system_spectral[2] = (system_spectral[2] * system_spectral[2] + system_spectral[3] * system_spectral[3]);
    system_spectral[2] %= 2;
    System53d system_lambert_cos2 = system_spectral.drop_equation(3);

    m_impl->pixel_size = m_impl->sensor_width / (float)width / m_impl->magnification;

    m_impl->blade_positions.resize(m_param.blade_count * 2);
#pragma omp parallel for
    for (int b = 0; b < m_param.blade_count; ++b) {
        m_impl->blade_positions[b * 2] = std::cos(b * 3.141596535f * 2.0f / m_param.blade_count) * m_impl->r_pupil;
        m_impl->blade_positions[b * 2 + 1] = std::sin(b * 3.141596535f * 2.0f / m_param.blade_count) * m_impl->r_pupil;
    }

#pragma omp parallel for
    m_impl->lambda_system.clear();
    for (int ll = 0; ll < m_param.num_lambdas; ++ll) {
        float lambda = m_param.lambda_from + (m_param.lambda_to - m_param.lambda_from) * (ll / (float)(m_param.num_lambdas - 1));
        if (m_param.num_lambdas == 1)
        {
            lambda = 550;
        }
        System43f system_lambda = system_lambert_cos2.bake_input_variable(4, lambda);
        system_lambda %= degree;
#pragma omp parallel for
        for (int j = 0; j < height; j++) 
        {
            const float y_sensor = ((j - height / 2) / (float)width) * m_param.sensor_width;
            const float y_world = y_sensor / m_impl->magnification;
            System33f system_y = system_lambda.bake_input_variable(1, y_world);
            m_impl->lambda_system.push_back(system_y);
        }
    }

    m_systems.resize(m_impl->lambda_system.size());
    for (int i = 0; i < m_impl->lambda_system.size(); i++) 
    {
        m_systems.at(i).trunc_degree = m_impl->lambda_system.at(i).trunc_degree;

        for (int j = 0; j < 3; j++) 
        {
            auto& cuda_equation = m_systems.at(i).equations[j];
            auto& equation = m_impl->lambda_system.at(i).equations[j];

            cuda_equation.consolidated = equation.consolidated;
            cuda_equation.trunc_degree = equation.trunc_degree;
            cuda_equation.term_count = (int32_t)equation.terms.size();

            for (int k = 0; k < cuda_equation.term_count; k++)
            {
                cuda_equation.terms[k].coefficient = equation.terms[k].coefficient;
                cuda_equation.terms[k].exponents[0] = (int32_t)equation.terms[k].exponents[0];
                cuda_equation.terms[k].exponents[1] = (int32_t)equation.terms[k].exponents[1];
                cuda_equation.terms[k].exponents[2] = (int32_t)equation.terms[k].exponents[2];
            }

        }
    }

    m_cuda_param.lambda_from = param.lambda_from;
    m_cuda_param.lambda_to = param.lambda_to;
    m_cuda_param.sensor_width = m_impl->sensor_width;
    m_cuda_param.magnification = m_impl->magnification;
    m_cuda_param.r_pupil = m_impl->r_pupil;
    m_cuda_param.pixel_size = m_impl->pixel_size;
    m_cuda_param.anamorphic = (float)param.anamorphic;
    m_cuda_param.sensor_scaling = m_impl->sensor_scaling;
    m_cuda_param.sensor_xres = (float)m_impl->sensor_xres;
    m_cuda_param.sensor_yres = (float)m_impl->sensor_yres;
    m_cuda_param.sample_mul = (float)param.sample_mul;
    m_cuda_param.exposure = (float)param.exposure;
    m_cuda_param.blade_coumt = (float)param.blade_count;
}

void LensSystem::CalcLensSystemImage(std::vector<float4>& src, std::vector<float4>& dst)
{
    std::stringstream ss;

    int blade_count = m_param.blade_count;
    int anamorphic = m_param.anamorphic;
    float sample_mul = m_param.sample_mul;
    int num_lambdas = m_param.num_lambdas;
    float exposure = m_param.exposure;
    std::string system_definition_file = m_param.system_definition_file.c_str();
    const float lambda_from = m_param.lambda_from;
    const float lambda_to = m_param.lambda_to;

    int width = m_param.width;
    int height = m_param.height;
    float sensor_width = m_impl->sensor_width;
    float magnification = m_impl->magnification;
    float r_pupil = m_impl->r_pupil;
    float pixel_size = m_impl->pixel_size;
    float sensor_scaling = m_impl->sensor_scaling;
    int sensor_xres = m_impl->sensor_xres;
    int sensor_yres = m_impl->sensor_yres;
    CImage& img_in = *m_impl->image_in.get();
    CImage& img_out = *m_impl->image_out.get();
    std::vector<float>& blade_positions = m_impl->blade_positions;
    std::vector<float>& rgb = m_impl->rgb;

    size_t input_byte_size = sizeof(float) * 4 * width * height;
    size_t output_byte_size = sizeof(float) * 3 * width * height;
    memcpy(img_in.data(), src.data(), input_byte_size);
    memset(img_out.data(), 0, output_byte_size);
#pragma omp parallel for
    for (int ll = 0; ll < num_lambdas; ++ll) {

        float lambda = lambda_from + (lambda_to - lambda_from) * (ll / (float)(num_lambdas - 1));

#pragma omp parallel for
        for (int j = 0; j < height; j++) {

            int index = ll * height + j;

            // Bake y dependency
            System33f& system_y = m_impl->lambda_system.at(index);
            
#pragma omp parallel for
            for (int i = 0; i < width; i++) {
                const float x_sensor = (i / (float)width - 0.5f) * sensor_width;
                const float x_world = x_sensor / magnification;

                // Sample intensity at wavelength lambda from source image
                const float rgbin[3] = {
                        exposure * img_in.linear_atXY((float)i, (float)j, 0, 0, 0.0f),
                        exposure * img_in.linear_atXY((float)i, (float)j, 0, 1, 0.0f),
                        exposure * img_in.linear_atXY((float)i, (float)j, 0, 2, 0.0f) };
                float L_in = spectrum_rgb_to_p(lambda, rgbin);

                // Quasi-importance sampling:
                // pick number of samples according to pixel intensity
                int num_samples = std::max(1, (int)(L_in * sample_mul));

                float sample_weight = L_in / num_samples;
#pragma omp parallel for
                // With that, we can now start sampling the aperture:
                for (int sample = 0; sample < num_samples; ++sample) {

                    // Rejection-sample points from lens aperture:
                    float x_ap, y_ap;

                    if (blade_count == 0) 
                    {
                        do {
                            x_ap = (rand() / (float)RAND_MAX - 0.5f) * 2.0f * r_pupil;
                            y_ap = (rand() / (float)RAND_MAX - 0.5f) * 2.0f * r_pupil;
                        } while (x_ap * x_ap + y_ap * y_ap > r_pupil * r_pupil);
                    }
                    else {
                        bool inside;
                        do {
                            inside = true;
                            x_ap = (rand() / (float)RAND_MAX - 0.5f) * 2.0f * r_pupil;
                            y_ap = (rand() / (float)RAND_MAX - 0.5f) * 2.0f * r_pupil;

                            for (int b = 0; b < blade_count; ++b) {
                                float bx = blade_positions[((b + 1) % blade_count) * 2] - blade_positions[b * 2];
                                float by =
                                    blade_positions[((b + 1) % blade_count) * 2 + 1] - blade_positions[b * 2 + 1];

                                float px = x_ap - blade_positions[b * 2];
                                float py = y_ap - blade_positions[b * 2 + 1];


                                float det = (px * by) - (py * bx); //bx * py - px * by;
                                if (det > 0) {

                                    inside = false;
                                    break;
                                }
                            }


                        } while (!inside);
                    }


                    float in[5], out[4];

                    // Fill in variables and evaluate systems:
                    in[0] = x_world + pixel_size * (rand() / (float)RAND_MAX - 0.5f);
                    in[1] = x_ap / anamorphic;
                    in[2] = y_ap;

                    system_y.evaluate(in, out);

                    // Scale to pixel size:
                    out[0] = out[0] * sensor_scaling + sensor_xres / 2;
                    out[1] = out[1] * sensor_scaling + sensor_yres / 2;

                    // out[2] contains one minus square of Lambertian cosine
                    float lambert = sqrt(1 - out[2]);
                    if (lambert != lambert) lambert = 0; // NaN check

                    float l =lambert * sample_weight;
                    img_out.set_linear_atXY(l * rgb[0 + 3 * ll], out[0], out[1], 0, 0, true);
                    img_out.set_linear_atXY(l * rgb[1 + 3 * ll], out[0], out[1], 0, 1, true);
                    img_out.set_linear_atXY(l * rgb[2 + 3 * ll], out[0], out[1], 0, 2, true);

                    //float l = L_in;
                    //img_out.set_linear_atXY(l * rgb[0 + 3 * ll], (float)i, (float)j, 0, 0, false);
                    //img_out.set_linear_atXY(l * rgb[1 + 3 * ll], (float)i, (float)j, 0, 1, false);
                    //img_out.set_linear_atXY(l * rgb[2 + 3 * ll], (float)i, (float)j, 0, 2, false);

                }
            }
        }
    }
#pragma omp parallel for
    // Fix gamut problem (pure wavelengths sometimes result in negative RGB)
    for (int j = 0; j < sensor_yres; ++j) {
#pragma omp parallel for
        for (int i = 0; i < sensor_xres; ++i) {
            float max_value = std::max(img_out.atXY(i, j, 0, 0),
                std::max(img_out.atXY(i, j, 0, 1),
                    img_out.atXY(i, j, 0, 2)));
            img_out.atXY(i, j, 0, 0) = std::max(img_out.atXY(i, j, 0, 0), 0.02f * max_value);
            img_out.atXY(i, j, 0, 1) = std::max(img_out.atXY(i, j, 0, 1), 0.02f * max_value);
            img_out.atXY(i, j, 0, 2) = std::max(img_out.atXY(i, j, 0, 2), 0.02f * max_value);

        }
    }

    std::vector<float3> tmp;
    tmp.resize(width* height);
    memcpy(tmp.data(), img_out.data(), output_byte_size);

#pragma omp parallel for
    for (int i = 0; i < dst.size(); i++) 
    {
        float3& value = tmp.at(i);
        dst.at(i) = make_float4(value.x, value.y, value.z, 1.0f);
    }
}

std::vector<float>& LensSystem::GetRGBParam()
{
    return m_impl->rgb;
}
std::vector<float>& LensSystem::GetBladePositions()
{
    return m_impl->blade_positions;
}
}