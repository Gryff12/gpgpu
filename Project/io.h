//
// Created by Maxime on 25/05/2023.
//

#ifndef PROJECT_IO_H
#define PROJECT_IO_H

#include <iostream>
#include <vector>
#include <png.h>

struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

Color **loadImage(const std::string &filename, unsigned &width, unsigned &height);

void saveImage(const char *filename, double **pixels, int width, int height);

void saveImage(const char *filename, bool *pixels, int width, int height);

void saveImage(const char *filename, Color *pixels, int width, int height);


#endif //PROJECT_IO_H
