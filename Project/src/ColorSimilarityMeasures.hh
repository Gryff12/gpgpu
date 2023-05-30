//
// Created by Nico on 25/05/2023.
//

#ifndef PROJECT_COLORSIMILARITYMEASURES_HH
#define PROJECT_COLORSIMILARITYMEASURES_HH

#include <vector>
#include "../io.h"

std::pair<double> **ColorSimilarityMeasures(Color **img1, uint8_t **img2, int width, int height);

#endif //PROJECT_COLORSIMILARITYMEASURES_HH
