#include "RCE3D.hpp"
#include "raytracer.cuh"

    RCE3D::RCE3D(int SCREENWIDTH, int SCREENHEIGHT) : SCREENWIDTH{SCREENWIDTH}, SCREENHEIGHT{SCREENHEIGHT} {
        windowTitle = "RCE3D";
    }

    bool RCE3D::start() {
        raytracer = new Raytracer(SCREENWIDTH, SCREENHEIGHT);
        setRelativeCursor(true);
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
        float* delta = (float*)malloc(5 * sizeof(float));
        memset(delta, 0.0, 5 * sizeof(float));
        if (getKeyState(SDLK_s).hold) {
            delta[2] += deltaTime * 2.0;
        } 
        if (getKeyState(SDLK_w).hold) {
            delta[2] -= deltaTime * 2.0;
        }
        if (getKeyState(SDLK_d).hold) {
            delta[0] += deltaTime * 1.0;
        } 
        if (getKeyState(SDLK_a).hold) {
            delta[0] -= deltaTime * 1.0;
        }
        if (getKeyState(SDLK_SPACE).hold) {
            delta[1] += deltaTime * 2.0;
        } 
        if (getKeyState(SDLK_c).hold) {
            delta[1] -= deltaTime * 2.0;
        }
        delta[3] += getRelCursorX() * deltaTime * 5.0;
        delta[4] += getRelCursorY() * deltaTime * 5.0;
        raytracer->update(screen, delta);
        delete delta;
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