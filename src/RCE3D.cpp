#include "RCE3D.hpp"
#include "raytracer.cuh"

    RCE3D::RCE3D(int SCREENWIDTH, int SCREENHEIGHT) : SCREENWIDTH{SCREENWIDTH}, SCREENHEIGHT{SCREENHEIGHT} {
        windowTitle = "RCE3D";
    }

    bool RCE3D::start() {
        raytracer = new Raytracer(SCREENWIDTH, SCREENHEIGHT);
        return true;
    }

    bool RCE3D::update(double deltaTime) {
        std::vector<std::vector<std::vector<int>>> screen;
        screen = std::vector<std::vector<std::vector<int>>>(SCREENHEIGHT);
        for (int i = 0; i < SCREENHEIGHT; i ++) {
            screen[i] = std::vector<std::vector<int>>(SCREENWIDTH);
            for (int j = 0; j < SCREENWIDTH; j ++) {
                screen[i][j] = std::vector<int>(3, 0);
            }
        }
        std::vector<float> delta = std::vector<float>(3, 0.0);
        raytracer->update(screen, delta);
        for (int i = 0; i < SCREENWIDTH; i ++) {
            for (int j = 0; j < SCREENHEIGHT; j ++) {
                draw(i, j, ' ', {0, 0, 0, 0}, {
                    static_cast<Uint8>(screen[j][i][0]),
                    static_cast<Uint8>(screen[j][i][1]),
                    static_cast<Uint8>(screen[j][i][2]),
                    255
                });
            }
        }
        
        return true;
    }

    bool RCE3D::destroy() {
        delete raytracer;
    
        return true;
    }