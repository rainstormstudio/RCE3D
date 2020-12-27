/**
 * @file sphere.hpp
 * @author Hongyu Ding
 * @brief This is the definition of sphere (CUDA)
 * @version 0.2
 * @date 2020-10-23
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "surface.cuh"
#include "vec3.cuh"

class Sphere : public Surface {
    point3 center;
    float radius;
public:
    Material *material;
    __device__ Sphere() {}
    __device__ Sphere(point3 cen, float r, Material *material) 
        : center{cen}, radius{r}, material{material} {}

    __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, Hit_record& rec) const override;
};

__device__ bool Sphere::hit(const Ray& ray, float t_min, float t_max, Hit_record& rec) const {
    vec3 oc = ray.origin() - center;
    auto a = dot(ray.direction(), ray.direction());
    auto b = dot(oc, ray.direction());
    auto c = dot(oc, oc) - radius * radius;
    auto discriminant = b * b - a * c;

    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = ray.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.material = material;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = ray.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.material = material;
            return true;
        }
    }
    return false;
}

#endif
