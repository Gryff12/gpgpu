//
// Created by Nico on 29/05/2023.
//

#include "Classification.hh"

void swap(double& a, double& b) {
    double temp = a;
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

double **IsBackgroundPixel(Color **img1, Color **img2, int width, int height, double threshold) {
	double **retVal = new double *[width];
    for (int i = 0; i < width; i++)
        retVal[i] = new double[height];

    double **textureSimilarity = TextureSimilarityMeasures(img1, img2, width, height);
    std::pair<double, double> **colorFeatures = ColorSimilarityMeasures(img1, img2, width, height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double x1 = textureSimilarity[x][y];
			double x2 = colorFeatures[x][y].first;
			double x3 = colorFeatures[x][y].second;

			sortThreeValues(x1, x2, x3);

			double scalar = 0.1 * x1 + 0.3 * x2 + 0.6 * x3;

            retVal[x][y] = scalar < threshold;
        }
    }
    return retVal;
}
