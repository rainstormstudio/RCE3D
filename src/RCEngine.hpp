/**
 * @file RCEngine.hpp
 * @author Daniel Hongyu Ding
 * @brief 
 * @version 0.1
 * @date 2020-12-25
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#pragma once
#ifndef RC_ENGINE_HPP
#define RC_ENGINE_HPP

#ifdef __linux__
#include "SDL2/SDL.h"
#include "SDL2/SDL_image.h"
#include "SDL2/SDL_mixer.h"
#include "SDL2/SDL_ttf.h"
#elif _WIN32
#include "SDL.h"
#include "SDL_image.h"
#include "SDL_mixer.h"
#include "SDL_ttf.h"
#endif

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

class CellTexture {
    SDL_Texture* texture;    // texture of the cell
    Uint8 ch;
    int numSrcRows;     // number of rows in src texture
    int numSrcCols;     // number of columns in src texture
    SDL_Rect srcRect;
    SDL_Rect destRect;
    SDL_Color foreColor;
    SDL_Color backColor;

public:
    /**
     * @brief Construct a new Cell 
     * 
     * @param texture the texture created from the tileset
     * @param numSrcRows number of rows in the tileset
     * @param numSrcCols number of columns in the tileset
     * @param srcCellWidth the width of the character in the tileset
     * @param srcCellHeight the height of the character in the tileset
     * @param destCellWidth the width of the cell
     * @param destCellHeight the height of the cell
     */
    CellTexture(SDL_Texture* texture, int numSrcRows, int numSrcCols, 
            int srcCellWidth, int srcCellHeight, int destCellWidth, int destCellHeight);

    /**
     * @brief Set the character of the cell
     * 
     * @param ch 
     */
    inline void setCh(Uint8 ch) {
        this->ch = ch;
        srcRect.x = (ch % numSrcCols) * srcRect.w;
        srcRect.y = (ch / numSrcCols) * srcRect.h;
    }

    /**
     * @brief Set the coordinate of the destRect
     * 
     * @param x x-coordinate
     * @param y y-coordinate
     */
    inline void setDestPosition(int x, int y) {
        destRect.x = x;
        destRect.y = y;
    }

    /**
     * @brief Set the coordinate of the srcRect
     * 
     * @param x 
     * @param y 
     */
    inline void setSrcPosition(int x, int y) {
        srcRect.x = x;
        srcRect.y = y;
    }

    /**
     * @brief Set the Fore Color of the cell
     * 
     * @param r red
     * @param g green
     * @param b blue
     * @param a alpha
     */
    inline void setForeColor(Uint8 r, Uint8 g, Uint8 b, Uint8 a) {
        foreColor = {r, g, b, a};
    }

    /**
     * @brief Set the Fore Color of the cell
     * 
     * @param color (r, g, b, a)
     */
    inline void setForeColor(SDL_Color color) {
        foreColor = color;
    }

    /**
     * @brief Set the Back Color of the cell
     * 
     * @param r red
     * @param g green
     * @param b blue
     * @param a alpha
     */
    inline void setBackColor(Uint8 r, Uint8 g, Uint8 b, Uint8 a) {
        backColor = {r, g, b, a};
    }

    /**
     * @brief Set the Back Color of the cell
     * 
     * @param color (r, g, b, a)
     */
    inline void setBackColor(SDL_Color color) {
        backColor = color;
    }

    /**
     * @brief Get the charactor of the cell
     * 
     * @return Uint8 
     */
    Uint8 getCh() const {
        return ch;
    }

    /**
     * @brief Get the Fore Color of the cell
     * 
     * @return SDL_Color 
     */
    SDL_Color getForeColor() const {
        return foreColor;
    }

    /**
     * @brief Get the Back Color of the cell
     * 
     * @return SDL_Color 
     */
    SDL_Color getBackColor() const {
        return backColor;
    }

    /**
     * @brief renders the cell to the screen
     * 
     * @param renderer 
     */
    void render(SDL_Renderer* renderer);
};

class RCEngine {
protected:
    // graphics info
    int cellRows;   // number of rows of cells
    int cellCols;   // number of columns of cells
    int cellWidth;  // the width of the cell
    int cellHeight; // the height of the cell
    int screenWidth;    // the width of the screen
    int screenHeight;   // the height of the screen
    std::string windowTitle; // the title of the window

    // inputs
    struct KeyState {
        bool pressed;
        bool released;
        bool hold;
    };
    std::vector<bool> keyInput;
    std::vector<bool> prevKeyInput;
    std::vector<KeyState> keyState;
    std::vector<bool> cursorInput;
    std::vector<bool> prevCursorInput;
    std::vector<KeyState> cursorState;
    int cursorPosX;
    int cursorPosY;
    int relCursorPosX;
    int relCursorPosY;

private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* tileset;
    std::vector<std::vector<std::shared_ptr<CellTexture>>> buffer;

    // events info
    SDL_Event event;

    // inputs
    const int TOTAL_KEYS = 332;
    const int TOTAL_CURSOR_STATES = 5;

    // game info
    bool loop;

public:
    RCEngine();

    ~RCEngine() {}

    /**
     * @brief Creates the console window
     * 
     * @param tilesetPath the path to the tileset
     * @param rows number of rows of cells
     * @param cols number of columns of cells
     * @param fontWidth the width of the cell
     * @param fontHeight the height of the cell
     * @return true
     * @return false
     */
    bool createConsole(std::string tilesetPath = "./RCE_tileset.png", int numSrcRows = 16, int numSrcCols = 16, int rows = 30, int cols = 40, int fontWidth = 20, int fontHeight = 20);
    
    /**
     * @brief blends two colors
     * 
     * @param color1 (r, g, b, a)
     * @param color2 (r, g, b, a)
     * @return SDL_Color 
     */
    SDL_Color blendColor(SDL_Color color1, SDL_Color color2);

    /**
     * @brief draws a charactor ch at (x, y) width foreColor and backColor
     * 
     * @param x x-coordinate (the index of column)
     * @param y y-coordinate (the index of row)
     * @param ch charactor
     * @param foreColor (r, g, b, a)
     * @param backColor (r, g, b, a)
     */
    void draw(int x, int y, Uint8 ch = ' ', SDL_Color foreColor = {255, 255, 255, 255}, SDL_Color backColor = {0, 0, 0, 255});

    /**
     * @brief draws a line from (x1, y1) to (x2, y2)
     * 
     * @param x1 
     * @param y1 
     * @param x2 
     * @param y2 
     * @param ch 
     * @param foreColor 
     * @param backColor 
     */
    void drawLine(int x1, int y1, int x2, int y2, Uint8 ch = ' ', SDL_Color foreColor = {255, 255, 255, 255}, SDL_Color backColor = {0, 0, 0, 255});

    /**
     * @brief Get the character at (x, y)
     * 
     * @param x x-coordinate (the index of column)
     * @param y y-coordinate (the index of row)
     * @return Uint8 
     */
    Uint8 getCh(int x, int y) const;

    /**
     * @brief Get the Fore Color at (x, y)
     * 
     * @param x x-coordinate (the index of column)
     * @param y y-coordinate (the index of row)
     * @return SDL_Color 
     */
    SDL_Color getForeColor(int x, int y) const;

    /**
     * @brief Get the Back Color at (x, y)
     * 
     * @param x x-coordinate (the index of column)
     * @param y y-coordinate (the index of row)
     * @return SDL_Color 
     */
    SDL_Color getBackColor(int x, int y) const;

    /**
     * @brief write a string to the screen starting at (x, y)
     * 
     * @param x x-coordinate (the index of column)
     * @param y y-coordinate (the index of row)
     * @param content 
     * @param foreColor (r, g, b, a)
     * @param backColor (r, g, b, a)
     */
    void write(int x, int y, std::string content, SDL_Color foreColor = {0, 0, 0, 0}, SDL_Color backColor = {0, 0, 0, 0});

    /**
     * @brief fills a rectangle region
     * 
     * @param dest (x, y, w, h)
     * @param ch character
     * @param foreColor (r, g, b, a)
     * @param backColor (r, g, b, a)
     */
    void fill(SDL_Rect dest, Uint8 ch = ' ', SDL_Color foreColor = {0, 0, 0, 0}, SDL_Color backColor = {0, 0, 0, 0});

    /**
     * @brief start the game loop
     * 
     */
    void init();

    /**
     * @brief clear screen and buffer
     * 
     */
    void clear();

    /**
     * @brief render to the screen
     * 
     */
    void render();
private:
    /**
     * @brief game loop
     * 
     */
    void gameLoop();

public:
    virtual bool start() = 0;
    virtual bool update(double deltaTime) = 0;
    virtual bool destroy() {
        return true;
    }

public:

    /**
     * @brief Get the x coordinate of the cursor
     * 
     * @return int 
     */
    int getCursorX() const {
        return cursorPosX;
    }

    /**
     * @brief Get the y coordinate of the cursor
     * 
     * @return int 
     */
    int getCursorY() const {
        return cursorPosY;
    }

    /**
     * @brief Get the relative motion of cursor
     * of x coordinate
     * 
     * @return int 
     */
    int getRelCursorX() const {
        return relCursorPosX;
    }

    /**
     * @brief Get the relative motion of cursor
     * of y coordinate
     * 
     * @return int 
     */
    int getRelCursorY() const {
        return relCursorPosY;
    }

    /**
     * @brief Set relative cursor mode
     * when it's on, the cursor is hidden and 
     * only relCursorPos will be active
     * 
     * @param value 
     */
    void setRelativeCursor(bool value) {
        if (value) {
            SDL_SetRelativeMouseMode(SDL_TRUE);
        } else {
            SDL_SetRelativeMouseMode(SDL_FALSE);
        }
    }

    /**
     * @brief Get the Key State of key
     * 
     * @param key 
     * @return KeyState 
     */
    KeyState getKeyState(int key) const {
        return keyState[key];
    }

    /**
     * @brief Get the Cursor State of cursor
     * 
     * @param cursor 
     * @return KeyState 
     */
    KeyState getCursorState(int cursor) const {
        return cursorState[cursor];
    }

protected:
    /**
     * @brief write debug message to standard error
     * 
     * @param msg 
     * @param level 
     */
    void debugMsg(std::string msg, int level = 0);

    /**
     * @brief write a line to standard error
     * 
     * @param level 
     */
    void debugLine(int level = 0);
};

#endif
