//
// Created by Nico on 25/05/2023.
//

#ifndef PROJECT_COLORSIMILARITYMEASURES_HH
#define PROJECT_COLORSIMILARITYMEASURES_HH

#include <vector>
#include "../io.h"

std::pair<double, double> **ColorSimilarityMeasures(Color **img1, Color **img2, int width, int height);

__global__ void ColorSimilarityKernel(Color *d_img1, Color *d_img2, double *d_colorFeatures_r, size_t pitch_r, double *d_colorFeatures_g, size_t pitch_g, int width, int height);

#endif //PROJECT_COLORSIMILARITYMEASURES_HH
