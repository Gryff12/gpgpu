//
// Created by Nico on 25/05/2023.
//

#ifndef PROJECT_COLORSIMILARITYMEASURES_HH
#define PROJECT_COLORSIMILARITYMEASURES_HH

#include <vector>
#include "../io.h"


#ifdef GPU
__global__ void ColorSimilarityKernel(ColorRG *d_img1, size_t pitch1, ColorRG *d_img2, size_t pitch2, double *d_colorFeatures_r, size_t pitch_r, double *d_colorFeatures_g, size_t pitch_g, int width, int height);
#else
std::pair<double, double> *ColorSimilarityMeasures(ColorRG *img1, ColorRG *img2, int width, int height);
#endif


#endif //PROJECT_COLORSIMILARITYMEASURES_HH
