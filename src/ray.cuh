/**
 * @file ray.hpp
 * @author Hongyu Ding
 * @brief This defines the Ray class 
 * (CUDA)
 * @version 0.2
 * @date 2020-10-23
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.cuh"

class Ray {
    point3 orig;
    vec3 dir;
public:
    __device__ Ray() {}
    __device__ Ray(const point3& origin, const vec3& direction)
        : orig{origin}, dir{direction} {}

    __device__ point3 origin() const { return orig; }
    __device__ vec3 direction() const { return dir; }

    __device__ point3 at(float t) const {
        return orig + t * dir;
    }
};

#endif
