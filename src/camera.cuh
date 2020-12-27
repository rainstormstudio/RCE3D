/**
 * @file camera.hpp
 * @author Hongyu Ding
 * @brief This file defines the Camera class (CUDA)
 * @version 0.2
 * @date 2020-10-24
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "utilities.cuh"

class Camera {
    point3 lookfrom;
    point3 lookat; 
    vec3 vup;
    float vfov;
    float aspect_ratio;
    float aperture;
    float focus_dist;

    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
    float time0, time1;

public:
    __device__ Camera() {
        lower_left_corner = vec3(-2.0, -1.0, -1.0);
        horizontal = vec3(4.0, 0.0, 0.0);
        vertical = vec3(0.0, 2.0, 0.0);
        origin = vec3(0.0, 0.0, 0.0);
    }
    __device__ Camera(
        point3 lookfrom, 
        point3 lookat, 
        vec3 vup,       // view up
        float vfov,    // vertical field of view
        float aspect_ratio,
        float aperture,
        float focus_dist,
        float t0 = 0.0,
        float t1 = 0.0
        ) : lookfrom{lookfrom}, lookat{lookat}, vup{vup}, vfov{vfov}, aspect_ratio{aspect_ratio}, aperture{aperture}, focus_dist{focus_dist} {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;
        
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist * w;

        lens_radius = aperture / 2;
        time0 = t0;
        time1 = t1;
    }

    __device__ void move(float* delta) {
        w = unit_vector(lookfrom - lookat);
        lookfrom += delta[2] * w;
        lookat += delta[2] * w;
        lookfrom.v[1] += delta[1];
        lookat.v[1] += delta[1];
        float x = lookfrom.v[0] + (lookat.v[0] - lookfrom.v[0]) * cos(delta[0]) - (lookat.v[2] - lookfrom.v[2]) * sin(delta[0]);
        float z = lookfrom.v[2] + (lookat.v[0] - lookfrom.v[0]) * sin(delta[0]) + (lookat.v[2] - lookfrom.v[2]) * cos(delta[0]);
        lookat.v[0] = x;
        lookat.v[2] = z;

        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;
        
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist * w;

    }

    __device__ Ray get_ray(float s, float t) const {
        return Ray(
            origin, 
            lower_left_corner + s * horizontal + t * vertical - origin);
    }
};

#endif
