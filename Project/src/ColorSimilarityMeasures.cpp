//
// Created by Nico on 25/05/2023.
//
#include "ColorSimilarityMeasures.hh"

std::vector<float> ColorSimilarityMeasures(const std::vector<unsigned char>& img1, const std::vector<unsigned char>& img2, int height, int width) {
    std::vector<float> similarities;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = (i * width + j) * 3; // Calcul de l'indice correspondant dans les vecteurs img1 et img2

            unsigned char green1 = img1[index + 1]; // Canal vert de l'image 1
            unsigned char green2 = img2[index + 1]; // Canal vert de l'image 2

            unsigned char red1 = img1[index]; // Canal rouge de l'image 1
            unsigned char red2 = img2[index]; // Canal rouge de l'image 2

            float minSum = static_cast<float>(std::min(green1, green2)) + static_cast<float>(std::min(red1, red2));
            float maxSum = static_cast<float>(std::max(green1, green2)) + static_cast<float>(std::max(red1, red2));

            float ratio = minSum / maxSum;
            similarities.push_back(ratio);
        }
    }

    return similarities;
}
