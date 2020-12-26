/**
 * @file surface_list.hpp
 * @author Hongyu Ding
 * @brief This is the definition of the list of surface (CUDA)
 * @version 0.2
 * @date 2020-10-23
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef SURFACE_LIST_HPP
#define SURFACE_LIST_HPP

#include "surface.cuh"
#include <memory>
#include <vector>

class Surface_list : public Surface {
    Surface** objects;
    int size;
public:
    __device__ Surface_list() {}
    __device__ Surface_list(Surface **object, int n) {
        objects = object;
        size = n;
    }
    
    __device__ virtual bool hit(const Ray &ray, float t_min, float t_max, Hit_record& rec) const override;

};

__device__ bool Surface_list::hit(const Ray &ray, float t_min, float t_max, Hit_record& rec) const {
    Hit_record temp_rec;
    bool hitted = false;
    auto closeest_so_far = t_max;

    for (int i = 0; i < size; i ++) {
        if (objects[i]->hit(ray, t_min, closeest_so_far, temp_rec)) {
            hitted = true;
            closeest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hitted;
}

#endif
