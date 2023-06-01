//
// Created by Nico on 29/05/2023.
//

#ifndef PROJECT_TEXTUREFEATURESEXTRACTION_HH
#define PROJECT_TEXTUREFEATURESEXTRACTION_HH

#include <vector>
#include "../io.h"

uint8_t getPixel(Color *image, int x, int y, int width, int height);

uint8_t *TextureFeaturesExtraction(Color *image, int width, int height);

#endif //PROJECT_TEXTUREFEATURESEXTRACTION_HH
