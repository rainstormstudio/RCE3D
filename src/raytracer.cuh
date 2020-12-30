#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <curand_kernel.h>
#include <vector>

class vec3;
class Camera;
class Surface;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
//#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

class Raytracer {
    void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

    const double aspect_ratio = 16.0 / 8.0;
    const int SAMPLES_PER_PIXEL = 20;
    const int MAX_DEPTH = 30;

    int num_pixels;
    size_t fb_size;
    const int block_width = 32;
    const int block_height = 32;

    Camera** d_camera;
    vec3 *fb;
    curandState *d_rand_state;
    Surface** d_list;
    Surface** d_world;
    float* camera_displacement;

    int SCREENWIDTH;
    int SCREENHEIGHT;

public:
    Raytracer(int SCREENWIDTH, int SCREENHEIGHT);

    ~Raytracer();

    void update(std::vector<std::vector<std::vector<int>>> &buffer, float* delta);
};

#endif
