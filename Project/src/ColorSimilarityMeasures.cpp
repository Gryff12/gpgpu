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


std::vector<float> ColorSimilarityMeasures(const std::vector<unsigned char>& img1, const std::vector<unsigned char>& img2, int height, int width) {
	std::vector<float> similarities;
	int numPixels = height * width;

	for (int i = 0; i < numPixels; i++) {
		int index = i * 3; // Index of the current pixel in the image vectors

		unsigned char green1 = img1[index + 1]; // Green channel of image 1
		unsigned char green2 = img2[index + 1]; // Green channel of image 2

		// Compute the similarity between the green channel values
		float similarity = static_cast<float>(green1) / static_cast<float>(green2);

		// Normalize the similarity value between 0 and 1
		similarity = std::min(std::max(similarity, 0.0f), 1.0f);

		similarities.push_back(similarity);
	}

	return similarities;
}