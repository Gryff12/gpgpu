//
// Created by Nico on 29/05/2023.
//

#include "TextureSimilarityMeasures.hh"

// Fonction pour convertir un unsigned char en unsigned int
unsigned int ToUnsignedInt(unsigned char value) {
    return static_cast<unsigned int>(value);
}

// Fonction pour calculer la similarité basée sur la distance de Hamming entre deux images
std::vector<float> TextureSimilarityMeasures(const std::vector<unsigned char>& img1, const std::vector<unsigned char>& img2, int height, int width) {
    std::vector<float> similarities;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = (i * width + j) * 3; // Calcul de l'indice correspondant dans les vecteurs img1 et img2

            unsigned char green1 = img1[index + 1]; // Canal vert de l'image 1
            unsigned char green2 = img2[index + 1]; // Canal vert de l'image 2

            unsigned char red1 = img1[index]; // Canal rouge de l'image 1
            unsigned char red2 = img2[index]; // Canal rouge de l'image 2

            unsigned int greenDistance = std::popcount(ToUnsignedInt(green1 ^ green2));
            unsigned int redDistance = std::popcount(ToUnsignedInt(red1 ^ red2));
            unsigned int distance = greenDistance + redDistance;

            float similarity = 1.0f - (static_cast<float>(distance) / 16); // Normalisation en divisant par le nombre total de bits

            similarities.push_back(similarity);
        }
    }

    return similarities;
}
