//
// Created by Nico on 29/05/2023.
//

#include "TextureSimilarityMeasures.hh"

// Fonction pour convertir un unsigned char en unsigned int
unsigned int ToUnsignedInt(unsigned char value) {
    return static_cast<unsigned int>(value);
}

// Fonction pour calculer la similarité basée sur la distance de Hamming entre deux images
double **TextureSimilarityMeasures(Color **img1, Color **img2, int width, int height) {
    double **similarities = new double *[width];
    for (int i = 0; i < width; i++)
        similarities[i] = new double[height];
    auto lbp1 = TextureFeaturesExtraction(img1, width, height);
    auto lbp2 = TextureFeaturesExtraction(img2, width, height);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
			uint8_t bitset1_2 = ToUnsignedInt(lbp1[x][y]);
			uint8_t bitset2_2 = ToUnsignedInt(lbp2[x][y]);
			uint8_t diff = bitset1_2 ^ bitset2_2;
			double count = std::popcount(diff);

            similarities[x][y] = (double) count / 8.;
        }
    }
    return similarities;
}
