#ifndef MATERIAL_CUH
#define MATERIAL_CUH

struct Hit_record;

#include "ray.cuh"
#include "surface.cuh"

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

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

class Dielectric : public Material {
    float ref_idx; // index of refraction

public:
    __device__ Dielectric(float index_of_refraction) : ref_idx{index_of_refraction} {}

    __device__ virtual bool scatter(const Ray& ray_in, const Hit_record &rec, color& attenuation, Ray& scattered, curandState *local_rand_state) const override {
        vec3 outward_normal;
        vec3 reflected = reflect(ray_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(ray_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(ray_in.direction(), rec.normal) / ray_in.direction().length();
            cosine = sqrt(1.0f - ref_idx * ref_idx * (1.0 - cosine * cosine));
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(ray_in.direction(), rec.normal) / ray_in.direction().length();
        }
        if (refract(ray_in.direction(), outward_normal, ni_over_nt, refracted)) {
            reflect_prob = schlick(cosine, ref_idx);
        } else {
            reflect_prob = 1.0f;
        }
        if (curand_uniform(local_rand_state) < reflect_prob) {
            scattered = Ray(rec.p, reflected);
        } else {
            scattered = Ray(rec.p, refracted);
        }
        return true;
    }
};

#endif
