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
    Material *material;
public:
    __device__ Sphere() {}
    __device__ Sphere(point3 cen, float r, Material *material) 
        : center{cen}, radius{r}, material{material} {}

    __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, Hit_record& rec) const override;
};

__device__ bool Sphere::hit(const Ray& ray, float t_min, float t_max, Hit_record& rec) const {
    vec3 oc = ray.origin() - center;
    auto a = ray.direction().length_squared();
    auto half_b = dot(oc, ray.direction());
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant > 0) {
        auto root = sqrt(discriminant);
        auto temp = (-half_b - root) / a;
        if (temp > t_min && temp < t_max) {
            rec.t = temp;
            rec.p = ray.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(ray, outward_normal);
            rec.material = material;
            return true;
        }

        temp = (-half_b + root) / a;
        if (temp > t_min && temp < t_max) {
            rec.t = temp;
            rec.p = ray.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(ray, outward_normal);
            rec.material = material;
            return true;
        }
    }
    return false;
}

#endif
