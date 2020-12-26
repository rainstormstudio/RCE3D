#ifndef RCE3D_HPP
#define RCE3D_HPP

#include "RCEngine.hpp"

class Raytracer;

class RCE3D : public RCEngine {
    int SCREENWIDTH;
    int SCREENHEIGHT;
    Raytracer* raytracer;
public:
    RCE3D(int SCREENWIDTH, int SCREENHEIGHT);

    bool start() override;

    bool update(double deltaTime) override;

    bool destroy() override;
};

#endif
