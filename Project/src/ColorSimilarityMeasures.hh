//
// Created by Nico on 25/05/2023.
//

#ifndef PROJECT_COLORSIMILARITYMEASURES_HH
#define PROJECT_COLORSIMILARITYMEASURES_HH

#include <vector>
#include "../io.h"

std::pair<double, double> **ColorSimilarityMeasures(Color **img1, Color **img2, int width, int height);

__global__ void ColorSimilarityKernel(Color *d_img1, Color *d_img2, std::pair<double, double> *d_colorFeatures, size_t pitch, int width, int height);

#endif //PROJECT_COLORSIMILARITYMEASURES_HH
