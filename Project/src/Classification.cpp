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

template<typename T>
void swap(T &a, T &b) {
    float temp = a;
    a = b;
    b = temp;
}

template<typename T>
void sortThreeValues(T &a, T &b, T &c) {
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
    double weights[3] = {0.1, 0.3, 0.6};

    // Tri des indicateurs par ordre croissant
    double sortedIndicators[3] = {x1, x2, x3};
    std::qsort(sortedIndicators, 3, sizeof(double),
               [](const void *x, const void *y) { return a - b; });

    // Calcul du produit scalaire avec le vecteur de pondération
    double weightedSum = weights[0] * sortedIndicators[0] + weights[1] * sortedIndicators[1] +
                         weights[2] * sortedIndicators[2];
    return weightedSum;
}

bool **IsBackgroundPixel(Color **img1, Color **img2, int width, int height, double threshold) {
    bool **retVal = new bool *[width];
    for (int i = 0; i < width; i++)
        retVal[i] = new bool[height];
    double **textureSimilarity = TextureSimilarityMeasures(img1, img2, width, height);
    std::pair<double> **colorFeatures = ColorSimilarityMeasures(img1, img2, width, height);
    for (int y = 0; i < height; i++) {
        for (int x = 0; j < width; j++) {
            double scalar = CalculateChoquetIntegral(textureSimilarity[x][y], colorFeatures[x][y].first,
                                                     colorFeatures[x][y].second);
            retVal[x][y] = scalar < threshold;
        }
    }
    return retVal;
}
