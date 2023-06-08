//
// Created by Nico on 29/05/2023.
//

#ifndef PROJECT_TEXTUREFEATURESEXTRACTION_HH
#define PROJECT_TEXTUREFEATURESEXTRACTION_HH

#include <vector>
#include "../io.h"

#ifndef GPU
uint8_t getPixel(ColorRG *image, int x, int y, int width, int height);
#endif

uint8_t *TextureFeaturesExtraction(ColorRG *image, int width, int height);

#endif //PROJECT_TEXTUREFEATURESEXTRACTION_HH
