#include "RCEngine.hpp"

    CellTexture::CellTexture(SDL_Texture* texture, int numSrcRows, int numSrcCols, 
            int srcCellWidth, int srcCellHeight, int destCellWidth, int destCellHeight)
    : texture{texture}, numSrcRows{numSrcRows}, numSrcCols{numSrcCols} {
        srcRect = {0, 0, srcCellWidth, srcCellHeight};
        destRect = {0, 0, destCellWidth, destCellHeight};
        foreColor = {255, 255, 255, 255};
        backColor = {0, 0, 0, 255};
        ch = 0;
    }

    void CellTexture::render(SDL_Renderer* renderer) {
        SDL_SetTextureColorMod(texture, backColor.r, backColor.g, backColor.b);
        SDL_SetTextureAlphaMod(texture, backColor.a);
        SDL_Rect backSrcRect = {static_cast<int>((219 % numSrcCols) * srcRect.w), static_cast<int>((219 / numSrcCols) * srcRect.h), srcRect.w, srcRect.h};
        SDL_RenderCopyEx(renderer, texture, &backSrcRect, &destRect, 0.0, nullptr, SDL_FLIP_NONE);
        SDL_SetTextureColorMod(texture, foreColor.r, foreColor.g, foreColor.b);
        SDL_SetTextureAlphaMod(texture, foreColor.a);
        SDL_RenderCopyEx(renderer, texture, &srcRect, &destRect, 0.0, nullptr, SDL_FLIP_NONE);
    }


    RCEngine::RCEngine() {
        cellRows = 0;
        cellCols = 0;
        screenWidth = 0;
        screenHeight = 0;
        windowTitle = "RCEngine";

        keyInput = std::vector<bool>(TOTAL_KEYS, false);
        prevKeyInput = std::vector<bool>(TOTAL_KEYS, false);
        keyState = std::vector<KeyState>(TOTAL_KEYS, {false, false, false});
        cursorInput = std::vector<bool>(TOTAL_CURSOR_STATES, false);
        prevCursorInput = std::vector<bool>(TOTAL_CURSOR_STATES, false);
        cursorState = std::vector<KeyState>(TOTAL_CURSOR_STATES, {false, false, false});
    }

    bool RCEngine::createConsole(std::string tilesetPath, int numSrcRows, int numSrcCols, int rows, int cols, int fontWidth, int fontHeight) {
        cellRows = rows;
        cellCols = cols;
        cellWidth = fontWidth;
        cellHeight = fontHeight;
        screenWidth = cellCols * cellWidth;
        screenHeight = cellRows * cellHeight;

        window = nullptr;
        renderer = nullptr;
        tileset = nullptr;

        loop = false;

        int tileWidth = 0;
        int tileHeight = 0;

        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_AUDIO) < 0) {
            std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
            return false;
        } else {
            window = SDL_CreateWindow(windowTitle.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, screenWidth, screenHeight, SDL_WINDOW_SHOWN);
            SDL_SetWindowFullscreen(window, 0);
            SDL_RaiseWindow(window);
            if (!window) {
                std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
                return false;
            } else {
                renderer = SDL_CreateRenderer(window, -1, 0);
                SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
                if (!(IMG_Init(IMG_INIT_PNG) & IMG_INIT_PNG)) {
                    std::cerr << "Failed to load SDL_image: " << IMG_GetError() << std::endl;
                    return false;
                }
                if (Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) < 0) {
                    std::cerr << "Failed to load SDL_mixer: " << Mix_GetError() << std::endl;
                    return false;
                }
            }
        }

        SDL_Surface* surface = IMG_Load(tilesetPath.c_str());
        if (!surface) {
            std::cerr << "Error initializing SDL surface: " << IMG_GetError() << std::endl;
            return false;
        } else {
            SDL_SetColorKey(surface, SDL_TRUE, SDL_MapRGB(surface->format, 255, 0, 255));
            tileset = SDL_CreateTextureFromSurface(renderer, surface);
            if (tileset == nullptr) {
                std::cerr << "Error creating texture from " << tilesetPath << ": " << SDL_GetError() << std::endl;
                return false;
            } else {
                tileWidth = surface->w / numSrcCols;
                tileHeight = surface->h / numSrcRows;
            }
            SDL_FreeSurface(surface);
        }

        buffer = std::vector<std::vector<std::shared_ptr<CellTexture>>>(cellRows);
        for (int i = 0; i < cellRows; i ++) {
            buffer[i] = std::vector<std::shared_ptr<CellTexture>>(cellCols);
            for (int j = 0; j < cellCols; j ++) {
                buffer[i][j] = std::make_shared<CellTexture>(tileset, numSrcRows, numSrcCols, tileWidth, tileHeight, cellWidth, cellHeight);
                buffer[i][j]->setDestPosition(j * cellWidth, i * cellHeight);
            }
        }
        return true;
    }

    SDL_Color RCEngine::blendColor(SDL_Color color1, SDL_Color color2) {
        double red = color1.r;
        double green = color1.g;
        double blue = color1.b;
        double alpha = color1.a;
        red = red * (alpha / 255.0);
        green = green * (alpha / 255.0);
        blue = blue * (alpha / 255.0);
        alpha = 0.0;
        double newRed = color2.r;
        double newGreen = color2.g;
        double newBlue = color2.b;
        double newAlpha = color2.a;
        red = newRed * (newAlpha / 255.0) + red * (1.0 - newAlpha / 255.0);
        green = newGreen * (newAlpha / 255.0) + green * (1.0 - newAlpha / 255.0);
        blue = newBlue * (newAlpha / 255.0) + blue * (1.0 - newAlpha / 255.0);
        alpha = 255.0;
        SDL_Color blended = {0, 0, 0, 0};
        blended.r = static_cast<Uint8>(round(red));
        blended.g = static_cast<Uint8>(round(green));
        blended.b = static_cast<Uint8>(round(blue));
        blended.a = static_cast<Uint8>(round(alpha));
        return blended;
    }

    void RCEngine::draw(int x, int y, Uint8 ch, SDL_Color foreColor, SDL_Color backColor) {
        if (0 <= x && x < cellCols && 0 <= y && y < cellRows) {
            buffer[y][x]->setCh(ch);
            buffer[y][x]->setForeColor(blendColor(buffer[y][x]->getForeColor(), foreColor));
            buffer[y][x]->setBackColor(blendColor(buffer[y][x]->getBackColor(), backColor));
        }
    }

    void RCEngine::drawLine(int x1, int y1, int x2, int y2, Uint8 ch, SDL_Color foreColor, SDL_Color backColor) {
        if (0 <= x1 && x1 < cellCols && 0 <= y1 && y1 < cellRows
            && 0 <= x2 && x2 < cellCols && 0 <= y2 && y2 < cellRows) {
            int dx = abs(x2 - x1);
            int dy = -abs(y2 - y1);

            // straight line optimization
            if (dx == 0) {  // vertical
                if (y1 > y2) {
                    std::swap(y1, y2);
                }
                for (int y = y1; y <= y2; y ++) {
                    draw(x1, y, ch, foreColor, backColor);
                }
            } else if (dy == 0) {   // horizontal
                if (x1 > x2) {
                    std::swap(x1, x2);
                }
                for (int x = x1; x <= x2; x ++) {
                    draw(x, y1, ch, foreColor, backColor);
                }
            } else {
                int sx = x1 < x2 ? 1 : -1;
                int sy = y1 < y2 ? 1 : -1;
                int error = dx + dy;
                while (1) {                    
                    draw(x1, y1, ch, foreColor, backColor);
                    if (x1 == x2 && y1 == y2) {
                        break;
                    }
                    int error2 = 2 * error;
                    if (error2 >= dy) {
                        error += dy;
                        x1 += sx;
                    }
                    if (error2 <= dx) {
                        error += dx;
                        y1 += sy;
                    }
                }
            }
        }
    }

    Uint8 RCEngine::getCh(int x, int y) const {
        if (0 <= x && x < cellCols && 0 <= y && y < cellRows) {
            return buffer[y][x]->getCh();
        } else {
            return 0;
        }
    }

    SDL_Color RCEngine::getForeColor(int x, int y) const {
        if (0 <= x && x < cellCols && 0 <= y && y < cellRows) {
            return buffer[y][x]->getForeColor();
        } else {
            return {0, 0, 0, 0};
        }
    }

    SDL_Color RCEngine::getBackColor(int x, int y) const {
        if (0 <= x && x < cellCols && 0 <= y && y < cellRows) {
            return buffer[y][x]->getBackColor();
        } else {
            return {0, 0, 0, 0};
        }
    }

    void RCEngine::write(int x, int y, std::string content, SDL_Color foreColor, SDL_Color backColor) {
        if (0 <= x && x < cellCols && 0 <= y && y < cellRows) {
            int len = content.length();
            for (int i = 0; i < len && x + i < cellCols; i ++) {
                if (content[i] == ' ') continue;
                buffer[y][x + i]->setCh(content[i]);
                buffer[y][x + i]->setForeColor(blendColor(buffer[y][x + i]->getForeColor(), foreColor));
                buffer[y][x + i]->setBackColor(blendColor(buffer[y][x + i]->getBackColor(), backColor));
            }
        }
    }

    void RCEngine::fill(SDL_Rect dest, Uint8 ch, SDL_Color foreColor, SDL_Color backColor) {
        if (0 <= dest.x && dest.x < cellCols && 0 <= dest.y && dest.y < cellRows) {
            for (int i = dest.y; i < dest.y + dest.h && i < cellRows; i ++) {
                for (int j = dest.x; j < dest.x + dest.w && j < cellCols; j ++) {
                    buffer[i][j]->setCh(ch);
                    buffer[i][j]->setForeColor(blendColor(buffer[i][j]->getForeColor(), foreColor));
                    buffer[i][j]->setBackColor(blendColor(buffer[i][j]->getBackColor(), backColor));
                }
            }
        }
    }

    void RCEngine::init() {
        loop = true;
        gameLoop();
    }

    void RCEngine::clear() {
        for (int i = 0; i < cellRows; i ++) {
            for (int j = 0; j < cellCols; j ++) {
                buffer[i][j]->setCh(' ');
                buffer[i][j]->setForeColor(255, 255, 255, 255);
                buffer[i][j]->setBackColor(0, 0, 0, 255);
            }
        }
        SDL_RenderClear(renderer);
    }

    void RCEngine::render() {
        for (int i = 0; i < cellRows; i ++) {
            for (int j = 0; j < cellCols; j ++) {
                buffer[i][j]->render(renderer);
            }
        }
        SDL_RenderPresent(renderer);
    }

    void RCEngine::gameLoop() {
        if (!start()) {
            loop = false;
        }

        auto time_a = std::chrono::high_resolution_clock::now();
        auto time_b = std::chrono::high_resolution_clock::now();

        while (loop) {
            while (loop) {
                time_b = std::chrono::high_resolution_clock::now();
                double deltaTime = std::chrono::duration_cast<std::chrono::microseconds>(time_b - time_a).count() / 1000000.0f;
                time_a = time_b;

                while (SDL_PollEvent(&event)) {
                    switch (event.type) {
                        case SDL_QUIT: {
                            loop = false;
                            break;
                        }
                        case SDL_KEYDOWN: {
                            keyInput[event.key.keysym.sym] = true;
                            break;
                        }
                        case SDL_KEYUP: {
                            keyInput[event.key.keysym.sym] = false;
                            break;
                        }
                        case SDL_MOUSEMOTION: {
                            cursorPosX = event.motion.x / cellWidth;
                            cursorPosY = event.motion.y / cellHeight;
                            break;
                        }
                        case SDL_MOUSEBUTTONDOWN: {
                            switch (event.button.button) {
                                case SDL_BUTTON_LEFT: {
                                    cursorInput[0] = true;
                                    break;
                                }
                                case SDL_BUTTON_RIGHT: {
                                    cursorInput[1] = true;
                                    break;
                                }
                                case SDL_BUTTON_MIDDLE: {
                                    cursorInput[2] = true;
                                    break;
                                }
                                case SDL_BUTTON_X1: {
                                    cursorInput[3] = true;
                                    break;
                                }
                                case SDL_BUTTON_X2: {
                                    cursorInput[4] = true;
                                    break;
                                }
                            }
                            break;
                        }
                        case SDL_MOUSEBUTTONUP: {
                            switch (event.button.button) {
                                case SDL_BUTTON_LEFT: {
                                    cursorInput[0] = false;
                                    break;
                                }
                                case SDL_BUTTON_RIGHT: {
                                    cursorInput[1] = false;
                                    break;
                                }
                                case SDL_BUTTON_MIDDLE: {
                                    cursorInput[2] = false;
                                    break;
                                }
                                case SDL_BUTTON_X1: {
                                    cursorInput[3] = false;
                                    break;
                                }
                                case SDL_BUTTON_X2: {
                                    cursorInput[4] = false;
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
                
                for (int i = 0; i < TOTAL_KEYS; i ++) {
                    keyState[i].pressed = false;
                    keyState[i].released = false;
                    if (keyInput[i] != prevKeyInput[i]) {
                        if (keyInput[i]) {
                            keyState[i].pressed = !keyState[i].hold;
                            keyState[i].hold = true;
                        } else {
                            keyState[i].released = true;
                            keyState[i].hold = false;
                        }
                    }
                    prevKeyInput[i] = keyInput[i];
                }

                for (int i = 0; i < TOTAL_CURSOR_STATES; i ++) {
                    cursorState[i].pressed = false;
                    cursorState[i].released = false;
                    if (cursorInput[i] != prevCursorInput[i]) {
                        if (cursorInput[i]) {
                            cursorState[i].pressed = true;
                            cursorState[i].hold = true;
                        } else {
                            cursorState[i].released = true;
                            cursorState[i].hold = false;
                        }
                    }
                    prevCursorInput[i] = cursorInput[i];
                }

                clear();

                if (!update(deltaTime)) {
                    loop = false;
                }
                std::string title = windowTitle + " - FPS: " + std::to_string(1.0f / deltaTime);
                SDL_SetWindowTitle(window, title.c_str());

                render();
            }

            if (destroy()) {
                if (!tileset) {
                    SDL_DestroyTexture(tileset);
                    tileset = nullptr;
                }
                SDL_DestroyWindow(window);
                SDL_DestroyRenderer(renderer);
                Mix_Quit();
                IMG_Quit();
                SDL_Quit();
            } else {
                loop = true;
            }
        }
    }

    void RCEngine::debugMsg(std::string msg, int level) {
        for (int i = 0; i < level; i ++) {
            std::cerr << "    ";
        }
        std::cerr << "| " << msg << std::endl;
    }

    void RCEngine::debugLine(int level) {
        for (int i = 0; i < level; i ++) {
            std::cerr << "    ";
        }
        for (int i = 0; i < 50; i ++) {
            std::cerr << "-";
        }
        std::cerr << std::endl;
    }
