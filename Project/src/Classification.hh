//
// Created by Nico on 29/05/2023.
//

#ifndef PROJECT_CLASSIFICATION_HH
#define PROJECT_CLASSIFICATION_HH


#include "../io.h"
#include "TextureFeaturesExtraction.hh"
#include "ColorSimilarityMeasures.hh"
#include "TextureSimilarityMeasures.hh"
#include <png.h>
#include <iostream>
#include <vector>
#include <cstdint>

double CalculateChoquetIntegral(double x1, double x2, double x3);

bool **IsBackgroundPixel(Color **img1, Color **img2, int width, int height, double threshold);

void sortThreeValues(double &a, double &b, double &c);

void swap(double &a, double &b);

__global__ void IsBackgroundPixelKernel(bool *d_retVal, double *d_textureSimilarity, std::pair<double, double> *d_colorFeatures, double threshold, int width, int height);

#endif //PROJECT_CLASSIFICATION_HH
