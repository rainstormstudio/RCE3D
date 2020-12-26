/**
 * @file vec3.hpp
 * @author Hongyu Ding
 * @brief This is the definition of vec3 class (CUDA)
 * @version 0.2
 * @date 2020-10-23
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#ifndef VEC3_CUH
#define VEC3_CUH

#include "utilities.cuh"
#include <cmath>
#include <iostream>

class vec3 {
public:
    float v[3];
    
    __host__ __device__ vec3() : v{0, 0, 0} {}
    __host__ __device__ vec3(float x, float y, float z) : v{x, y, z} {}

    __host__ __device__ inline float x() const { return v[0]; }
    __host__ __device__ inline float y() const { return v[1]; }
    __host__ __device__ inline float z() const { return v[2]; }
    __host__ __device__ inline float r() const { return v[0]; }
    __host__ __device__ inline float g() const { return v[1]; }
    __host__ __device__ inline float b() const { return v[2]; }

    __host__ __device__ inline vec3 operator-() const { return vec3(-v[0], -v[1], -v[2]); }
    __host__ __device__ inline float operator[](int i) const { return v[i]; }
    __host__ __device__ inline float& operator[](int i) { return v[i]; }

    __host__ __device__ inline vec3& operator+=(const vec3 &vec) {
        v[0] += vec.v[0];
        v[1] += vec.v[1];
        v[2] += vec.v[2];
        return *this;
    }

    __host__ __device__ inline vec3& operator-=(const vec3 &vec) {
        v[0] -= vec.v[0];
        v[1] -= vec.v[1];
        v[2] -= vec.v[2];
        return *this;
    }

    __host__ __device__ inline vec3& operator*=(const vec3 &vec) {
        v[0] *= vec.v[0];
        v[1] *= vec.v[1];
        v[2] *= vec.v[2];
        return *this;
    }

    __host__ __device__ inline vec3& operator*=(const float k) {
        v[0] *= k;
        v[1] *= k;
        v[2] *= k;
        return *this;
    }

    __host__ __device__ inline vec3& operator/=(const float k) {
        return *this *= 1/k;
    }

    __host__ __device__ inline float length() const {
        return std::sqrt(length_squared());
    }

    __host__ __device__ inline float length_squared() const {
        return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    }
};

// Type aliases for vec3
using point3 = vec3;
using color = vec3;

// Utility functions

inline std::ostream& operator<<(std::ostream &out, const vec3 &value) {
    return out << value.x() << " " << value.y() << " " << value.z();
}

__host__ __device__ inline vec3 operator+(const vec3 &a, const vec3 &b) {
    return vec3(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
}

__host__ __device__ inline vec3 operator-(const vec3 &a, const vec3 &b) {
    return vec3(a.x() - b.x(), a.y() - b.y(), a.z() - b.z());
}

__host__ __device__ inline vec3 operator*(const vec3 &a, const vec3 &b) {
    return vec3(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
}

__host__ __device__ inline vec3 operator*(float k, const vec3 &a) {
    return vec3(a.x() * k, a.y() * k, a.z() * k);
}

__host__ __device__ inline vec3 operator*(const vec3 &a, float k) {
    return k * a;
}

__host__ __device__ inline vec3 operator/(const vec3 &a, float k) {
    return (1/k) * a;
}

__host__ __device__ inline float dot(const vec3 &a, const vec3 &b) {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

__host__ __device__ inline vec3 cross(const vec3 &a, const vec3 &b) {
    return vec3(a.y() * b.z() - a.z() * b.y(),
                a.z() * b.x() - a.x() * b.z(),
                a.x() * b.y() - a.y() * b.x());
}

__host__ __device__ inline vec3 unit_vector(const vec3 &v) {
    return v / v.length();
}
/*
__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v, n) * n;
}

__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    auto cos_theta = dot(-uv, n);
    vec3 ray_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 ray_out_parallel = -sqrt(fabs(1.0 - ray_out_perp.length_squared())) * n;
    return ray_out_perp + ray_out_parallel;
}*/

#endif
