//
// Created by Nico on 29/05/2023.
//

#ifndef PROJECT_CLASSIFICATION_HH
#define PROJECT_CLASSIFICATION_HH

#include <vector>
#include <cstdint>

double** classifactionIndicators(int width, int height, uint8_t** lbpCode, double** colorSimilarityMeasures, double** textureSimilarityMeasures);


void saveImage(const char *filename, double** pixels, int width, int height);


#endif //PROJECT_CLASSIFICATION_HH
