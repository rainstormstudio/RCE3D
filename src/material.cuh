#ifndef MATERIAL_CUH
#define MATERIAL_CUH

struct Hit_record;

#include "ray.cuh"
#include "surface.cuh"

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2.0f * dot(v, n) * n;
}

class Material {
public:
    __device__ virtual bool scatter(const Ray &ray_in, const Hit_record &rec, color& attenuation, Ray& scattered, curandState *local_rand_state) const = 0;
};

class Diffuse : public Material {
    color albedo;
public:
    __device__ Diffuse(const color& a) : albedo{a} {}

    __device__ virtual bool scatter(const Ray &ray_in, const Hit_record &rec, color& attenuation, Ray& scattered, curandState *local_rand_state) const override {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = Ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }
};

class Metal : public Material {
    color albedo;
    float fuzz;
public:
    __device__ Metal(const color& a, float fuzz) : albedo{a}, fuzz{fuzz < 1 ? fuzz : 1} {}

    __device__ virtual bool scatter(const Ray &ray_in, const Hit_record &rec, color& attenuation, Ray& scattered, curandState *local_rand_state) const override {
        vec3 reflected = reflect(unit_vector(ray_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
};
/*
class Dielectric : public Material {
    float ir; // index of refraction

    static float reflectance(float cosine, float ref_idx) {
        // Schlick's approximation for reflectance
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }

public:
    __device__ Dielectric(float index_of_refraction) : ir{index_of_refraction} {}

    __device__ virtual bool scatter(const Ray& ray_in, const Hit_record &rec, color& attenuation, Ray& scattered, curandState *local_rand_state) const {
        attenuation = color(1.0, 1.0, 1.0);
        float refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        vec3 unit_direction = unit_vector(ray_in.direction());
        float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float()) {
            direction = reflect(unit_direction, rec.normal);
        } else {
            direction = refract(unit_direction, rec.normal, refraction_ratio);
        }

        scattered = Ray(rec.p, direction, ray_in.getTime());
        return true;
    }
};
*/
#endif
