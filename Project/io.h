//
// Created by Maxime on 25/05/2023.
//

#ifndef PROJECT_IO_H
#define PROJECT_IO_H

#include <iostream>
#include <vector>
#include <png.h>

bool loadImage(const std::string& filename, std::vector<unsigned char>& image, unsigned& width, unsigned& height);

#endif //PROJECT_IO_H
