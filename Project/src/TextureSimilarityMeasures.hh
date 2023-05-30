//
// Created by Nico on 29/05/2023.
//

#ifndef PROJECT_TEXTURESIMILARITYMEASURES_HH
#define PROJECT_TEXTURESIMILARITYMEASURES_HH


#include <vector>
#include <bit>
#include <bitset>
#include <cstdint>
#include "../io.h"
#include "textureFeaturesExtraction.hh"

double** TextureSimilarityMeasures(Color** img1, Color** img2, int width, int height);
#endif //PROJECT_TEXTURESIMILARITYMEASURES_HH
