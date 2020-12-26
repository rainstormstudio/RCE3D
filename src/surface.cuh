/**
 * @file surface.hpp
 * @author Hongyu Ding
 * @brief This is the definition of Surface class (CUDA)
 * @version 0.2
 * @date 2020-10-23
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef SURFACE_CUH
#define SURFACE_CUH

#include "utilities.cuh"
#include "ray.cuh"

class Material;

struct Hit_record {
    point3 p;
    vec3 normal;
    Material** material;
    float t;
    bool front_face;

    __device__ inline void set_face_normal(const Ray& ray, const vec3& outward_normal) {
        front_face = dot(ray.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Surface { // hittable
public:
    __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, Hit_record& rec) const = 0;
};

#endif
