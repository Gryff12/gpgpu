//
// Created by Nico on 29/05/2023.
//

#include "Classification.hh"
#include "../io.h"
#include "TextureFeaturesExtraction.hh"
#include "ColorFeaturesExtraction.hh"
#include "ColorSimilarityMeasures.hh"
#include "TextureSimilarityMeasures.hh"
#include <png.h>
#include <iostream>

template <typename T>
void swap(T& a, T& b) {
    float temp = a;
    a = b;
    b = temp;
}

template <typename T>
void sortThreeValues(T& a, T& b, T& c) {
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

void saveImage(const char *filename, double** pixels, int width, int height) {
	FILE *f = fopen(filename, "w");
	(void) fprintf(f, "P6\n%d %d\n255\n", width, height);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; ++x) {
			static unsigned char color[3];
			color[0] = (unsigned char) (std::min(1., pixels[x][y]) * 255);
			color[1] = (unsigned char) (std::min(1., pixels[x][y]) * 255);
			color[2] = (unsigned char) (std::min(1., pixels[x][y]) * 255);

			(void) fwrite(color, 1, 3, f);
		}
	}
	fclose(f);
}

//double** classifactionIndicators(int width, int height, uint8_t** lbpCode, double** colorSimilarityMeasures, double** textureSimilarityMeasures) {
//
//	double** classificationIndicators = new double*[width];
//	for (int i = 0; i < width; i++)
//		classificationIndicators[i] = new double[height];
//
//	for (int i = 0; i < width; i++) {
//		for (int j = 0; j < height; j++) {
//			double a = static_cast<double>(lbpCode[i][j]) / 255.;
//			double b = colorSimilarityMeasures[i][j];
//			double c = textureSimilarityMeasures[i][j];
//
//			//std::cout << "a: " << a << " b: " << b << " c: " << c << std::endl;
//
//			//Order elements in ascending order
//			sortThreeValues(a, b, c);
//			double raw_value = a * 0.1 + b * 0.3 + c * 0.6;
//			double value = raw_value > 0.67 ? 0 : 1;
//			classificationIndicators[i][j] = value;
//
//		}
//	}
//
//    return classificationIndicators;
//}



double CalculateChoquetIntegral(double x1, double x2, double x3) {
	double weights[3] = { 0.1, 0.3, 0.6 };

	// Tri des indicateurs par ordre croissant
	double sortedIndicators[3] = { x1, x2, x3 };
	std::sort(sortedIndicators, sortedIndicators + 3);

	// Calcul du produit scalaire avec le vecteur de pondération
	double weightedSum = weights[0] * sortedIndicators[0] + weights[1] * sortedIndicators[1] + weights[2] * sortedIndicators[2];

	return weightedSum;
}


bool IsBackgroundPixel(Color** img, int width, int height) {
	// Extraction des fonctionnalités de texture
	auto textureFeatures = TextureFeaturesExtraction(img, width, height);

	// Extraction des fonctionnalités de couleur
	auto colorFeatures = ColorFeaturesExtraction(img, width, height);

	// Calcul des similarités dans l'espace couleur
	double** colorSimilarity = ColorSimilarityMeasures(img, colorFeatures, width, height);

	bool isBackground = true;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			// Calcul de la similarité dans l'espace texture
			double textureSimilarity = TextureSimilarityMeasures(img, textureFeatures, width, height);

			// Calcul des indicateurs
			Indicators indicators;
			indicators.x1 = colorSimilarity[i][j];
			indicators.x2 = colorSimilarity[i][j];
			indicators.x3 = textureSimilarity;

			// Calcul du Choquet Integral
			double scalarValue = CalculateChoquetIntegral(indicators.x1, indicators.x2, indicators.x3);

			// Classification en fonction du seuil
			if (scalarValue >= 0.67) {
				isBackground = false;
				break;
			}
		}
		if (!isBackground) {
			break;
		}
	}

	return isBackground;
}