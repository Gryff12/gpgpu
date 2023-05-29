//
// Created by Nico on 29/05/2023.
//

#ifndef PROJECT_CLASSIFICATION_HH
#define PROJECT_CLASSIFICATION_HH

#include <vector>
#include <cstdint>

std::vector<float> classifactionIndicators(std::vector<uint8_t> lbpCode, std::vector<float> colorSimilarityMeasures, std::vector<float> textureSimilarityMeasures);



#endif //PROJECT_CLASSIFICATION_HH
