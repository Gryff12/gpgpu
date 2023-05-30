//
// Created by Nico on 29/05/2023.
//

#include "Classification.hh"

void swap(double& a, double& b) {
    int temp = a;
    a = b;
    b = temp;
}

void sortThreeValues(double& a, double& b, double& c) {
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

double CalculateChoquetIntegral(double x1, double x2, double x3) {
    double weights[3] = {0.1, 0.3, 0.6};

    // Tri des indicateurs par ordre croissant
    sortThreeValues(x1, x2, x3);

    // Calcul du produit scalaire avec le vecteur de pondération
    double weightedSum = weights[0] * x1 + weights[1] * x2 +
                         weights[2] * x3;
    return weightedSum;
}

bool **IsBackgroundPixel(Color **img1, Color **img2, int width, int height, double threshold) {
    bool **retVal = new bool *[width];
    for (int i = 0; i < width; i++)
        retVal[i] = new bool[height];
    double **textureSimilarity = TextureSimilarityMeasures(img1, img2, width, height);
    std::pair<double, double> **colorFeatures = ColorSimilarityMeasures(img1, img2, width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double scalar = CalculateChoquetIntegral(textureSimilarity[x][y], colorFeatures[x][y].first,
                                                     colorFeatures[x][y].second);
            retVal[x][y] = scalar < threshold;
        }
    }
    return retVal;
}
