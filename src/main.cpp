#include "RCE3D.hpp"

const int SCREENWIDTH = 80 * 3;
const int SCREENHEIGHT = 40 * 3;

int main() {
    RCE3D app(SCREENWIDTH, SCREENHEIGHT);
    if (app.createConsole("./assets/RCE_tileset.png", 16, 16, SCREENHEIGHT, SCREENWIDTH, 5, 5)) {
        app.init();
    }
    return 0;
}