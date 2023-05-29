//
// Created by Nico on 29/05/2023.
//

#include "Classification.hh"

void swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
}

void sortThreeValues(float& a, float& b, float& c) {
    if (a > b) {
        swap(a, b);
    }

    if (a > c) {
        swap(a, c);
    }

    if (b > c) {
        swap(b, c);
    }
}

std::vector<float> classifactionIndicators(std::vector<uint8_t> lbpCode, std::vector<float> colorSimilarityMeasures, std::vector<float> textureSimilarityMeasures){
    std::vector<float> classificationIndicators;
    for (int i = 0; i < lbpCode.size(); i++){
        float a = static_cast<float>(lbpCode[i]);
        float b = colorSimilarityMeasures[i];
        float c = textureSimilarityMeasures[i];

        //Order elements in ascending order
        sortThreeValues(a, b, c);
        float value = a * 0.1 + b * 0.3 + c * 0.6;
        classificationIndicators.push_back(value < 0.67 ? 0 : 1);
    }
    return classificationIndicators;
}



