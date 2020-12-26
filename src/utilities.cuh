/**
 * @file utilities.hpp
 * @author Hongyu Ding
 * @brief This file defines some constants and utility functions 
 * (CUDA)
 * @version 0.2
 * @date 2020-10-23
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef UTILITIES_CUH
#define UTILITIES_CUH

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
#include <time.h>
#include <chrono>

// Constants
const float INF = std::numeric_limits<float>::infinity();
const float PI = 3.1415926535897932384626;

// utility functions

__host__ __device__ inline 
float degrees_to_radians(float degrees) {
    return degrees * PI / 180.0;
}

__host__ __device__ inline 
float clamp(float  x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Common Headers
//#include "ray.hpp"
#include "vec3.cuh"

#endif
