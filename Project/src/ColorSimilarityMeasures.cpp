//
// Created by Nico on 25/05/2023.
//
#include "ColorSimilarityMeasures.hh"

//std::vector<float> ColorSimilarityMeasures(const std::vector<unsigned char>& img1, const std::vector<unsigned char>& img2, int height, int width) {
//    std::vector<float> similarities;
//    for (int i = 0; i < height; i++) {
//        for (int j = 0; j < width; j++) {
//            int index = (i * width + j) * 3; // Calcul de l'indice correspondant dans les vecteurs img1 et img2
//
//            unsigned char green1 = img1[index + 1]; // Canal vert de l'image 1
//            unsigned char green2 = img2[index + 1]; // Canal vert de l'image 2
//
//            unsigned char red1 = img1[index]; // Canal rouge de l'image 1
//            unsigned char red2 = img2[index]; // Canal rouge de l'image 2
//
//            float minSum = static_cast<float>(std::min(green1, green2)) + static_cast<float>(std::min(red1, red2));
//            float maxSum = static_cast<float>(std::max(green1, green2)) + static_cast<float>(std::max(red1, red2));
//
//			float ratio = minSum / maxSum;
//			if (maxSum == 0)
//				ratio = 0.f;
//
//            similarities.push_back(ratio);
//        }
//    }
//
//    return similarities;
//}


double** ColorSimilarityMeasures(Color** img1, uint8_t ** img2, int width, int height) {
	double** similarityMap = new double*[width];

	for (int i = 0; i < width; i++) {
		similarityMap[i] = new double[height];
		for (int j = 0; j < height; j++) {
			// Obtaining the color components of the current pixel
			uint8_t r1 = img1[i][j].r;
			uint8_t g1 = img1[i][j].g;
			//uint8_t b1 = img1[i][j].b;

			// Obtaining the color components of the background model pixel
			uint8_t r2 = img2[i][j];
			uint8_t g2 = r2;
			//uint8_t b2 = img2[i][j].b;

			// Computing the similarity measures for each color component
			double similarityR = 1.0 - (abs(r1 - r2) / 255.0);
			double similarityG = 1.0 - (abs(g1 - g2) / 255.0);
			//double similarityB = 1.0 - (abs(b1 - b2) / 255.0);

			// Taking the average similarity measure for each component
			similarityMap[i][j] = (similarityR + similarityG) / 2.0;
		}
	}

	return similarityMap;
}