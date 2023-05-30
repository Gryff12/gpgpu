//
// Created by Nico on 29/05/2023.
//

#include "TextureSimilarityMeasures.hh"

// Fonction pour convertir un unsigned char en unsigned int
unsigned int ToUnsignedInt(unsigned char value) {
    return static_cast<unsigned int>(value);
}

// Fonction pour calculer la similarité basée sur la distance de Hamming entre deux images
double** TextureSimilarityMeasures(Color** img1, Color** img2, int width, int height) {
	double** similarities = new double*[width];
	for (int i = 0; i < width; i++)
		similarities[i] = new double[height];

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {

            uint8_t green1 = img1[x][y].g; // Canal vert de l'image 1
			uint8_t green2 = img2[x][y].g; // Canal vert de l'image 2

			uint8_t red1 = img1[x][y].r; // Canal rouge de l'image 1
			uint8_t red2 = img2[x][y].r; // Canal rouge de l'image 2


			uint8_t greenDistance = std::popcount(ToUnsignedInt(green1 ^ green2));
			uint8_t redDistance = std::popcount(ToUnsignedInt(red1 ^ red2));
			uint8_t distance = greenDistance + redDistance;

            double similarity = 1.0 - (static_cast<double>(distance) / 16.); // Normalisation en divisant par le nombre total de bits

			similarities[x][y] = similarity;
        }
    }

    return similarities;
}
