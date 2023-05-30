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

#endif //PROJECT_CLASSIFICATION_HH
