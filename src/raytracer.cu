#include "raytracer.cuh"
#include <float.h>
#include "vec3.cuh"
#include "ray.cuh"
#include "surface.cuh"
#include "surface_list.cuh"
#include "sphere.cuh"
#include "camera.cuh"

    void Raytracer::check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
        if (result) {
            std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                    file << ":" << line << " '" << func << "' \n";
    
            cudaDeviceReset();
            exit(99);
        }
    }

    __device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
        vec3 p;
        do {
            p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
        } while (p.length_squared() >= 1.0f);
        return p;
    }
    
    __device__ color ray_trace(const Ray& ray, Surface** world, int max_depth, curandState* local_rand_state) {
        Ray current_ray = ray;
        float current_attenuation = 1.0f;
        for (int i = 0; i < max_depth; i ++) {
            Hit_record rec;
            if ((*world)->hit(current_ray, 0.0001, FLT_MAX, rec)) {
                vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
                current_attenuation *= 0.5f;
                current_ray = Ray(rec.p, target - rec.p);
            } else {
                vec3 unit_direction = unit_vector(current_ray.direction());
                auto t = 0.5f * (unit_direction.y() + 1.0f);
                color c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
                return current_attenuation * c;
            }
        }
        return vec3(0.0, 0.0, 0.0);
    }
    
    __global__ void render_init(int max_x, int max_y, curandState* rand_state) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if ((i >= max_x) || (j >= max_y)) return;
        int pixel_index = j * max_x + i;
    
        curand_init(clock64(), pixel_index, 0, &rand_state[pixel_index]);
    }
    
    __global__ void render(vec3 *fb, int max_x, int max_y, int max_depth, int samples_per_pixel, Camera** camera, Surface** world, curandState* rand_state) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if ((i >= max_x) || (j >= max_y)) return;
        int pixel_index = j * max_x + i;
        curandState local_rand_state = rand_state[pixel_index];
        color pixel_color(0, 0, 0);
        for (int sample = 0; sample < samples_per_pixel; sample ++) {
            float u = static_cast<float>(i + curand_uniform(&local_rand_state)) / static_cast<float>(max_x);
            float v = static_cast<float>(j + curand_uniform(&local_rand_state)) / static_cast<float>(max_y);
            Ray ray = (*camera)->get_ray(u, v);
            pixel_color += ray_trace(ray, world, max_depth, &local_rand_state);
        }
        rand_state[pixel_index] = local_rand_state;
        pixel_color /= static_cast<float>(samples_per_pixel);
        pixel_color[0] = sqrt(pixel_color[0]);
        pixel_color[1] = sqrt(pixel_color[1]);
        pixel_color[2] = sqrt(pixel_color[2]);
        fb[pixel_index] = pixel_color;
    }
    
    __global__ void create_scene(Surface** d_list, Surface** d_world, Camera** d_camera) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *(d_list + 0) = new Sphere(vec3(0, 0, -1), 0.5);
            *(d_list + 1) = new Sphere(vec3(0, -100.5, -1), 100);
            *d_world = new Surface_list(d_list, 2);
            *d_camera = new Camera();
        }
    }
    
    __global__ void free_scene(Surface** d_list, Surface** d_world, Camera** d_camera) {
        delete *(d_list + 0);
        delete *(d_list + 1);
        delete *(d_world);
        delete *(d_camera);
    }

    Raytracer::Raytracer(int SCREENWIDTH, int SCREENHEIGHT) : SCREENWIDTH(SCREENWIDTH), SCREENHEIGHT(SCREENHEIGHT){
        // GPU settings
        int num_pixels = SCREENWIDTH * SCREENHEIGHT;
        size_t fb_size = num_pixels * sizeof(vec3);   // (r, g, b)

        // camera
        checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera*)));

        // allocate framebuffer
        checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

        // allocate random state
        checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

        // create scene
        checkCudaErrors(cudaMalloc((void**)&d_list, 2*sizeof(Surface*)));
        checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Surface*)));
        create_scene<<<1, 1>>>(d_list, d_world, d_camera);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    Raytracer::~Raytracer() {
        checkCudaErrors(cudaDeviceSynchronize());
        free_scene<<<1, 1>>>(d_list, d_world, d_camera);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaFree(d_list));
        checkCudaErrors(cudaFree(d_world));
        checkCudaErrors(cudaFree(fb));
    
        cudaDeviceReset();
    }

    void Raytracer::update(std::vector<std::vector<std::vector<int>>> &buffer, std::vector<float> delta) {
        dim3 blocks(SCREENWIDTH / block_width + 1, SCREENHEIGHT / block_height + 1);
        dim3 threads(block_width, block_height);
        render_init<<<blocks, threads>>>(SCREENWIDTH, SCREENHEIGHT, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        render<<<blocks, threads>>>(fb, SCREENWIDTH, SCREENHEIGHT, MAX_DEPTH, SAMPLES_PER_PIXEL, d_camera, d_world, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        
        for (int j = SCREENHEIGHT - 1; j >= 0; j --) {
            for (int i = 0; i < SCREENWIDTH; i ++) {
                size_t pixel_index = j * SCREENWIDTH + i;
                auto r = fb[pixel_index].r();
                auto g = fb[pixel_index].g();
                auto b = fb[pixel_index].b();
                int ir = static_cast<int>(255.999 * r);
                int ig = static_cast<int>(255.999 * g);
                int ib = static_cast<int>(255.999 * b);
                buffer[SCREENHEIGHT - 1 - j][i][0] = ir;
                buffer[SCREENHEIGHT - 1 - j][i][1] = ig;
                buffer[SCREENHEIGHT - 1 - j][i][2] = ib;
            }
        }
    }