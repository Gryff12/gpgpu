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

bool *IsBackgroundPixel(Color *img1, Color *img2, int width, int height, double threshold) {
	bool *retVal = new bool [width * height];

    double *textureSimilarity = TextureSimilarityMeasures(img1, img2, width, height);
    std::pair<double, double> *colorFeatures = ColorSimilarityMeasures(img1, img2, width, height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double x1 = (double) textureSimilarity[y * width + x];
			double x2 = colorFeatures[y * width + x].first;
			double x3 = colorFeatures[y * width + x].second;

			sortThreeValues(x1, x2, x3);

			double scalar = 0.1 * x1 + 0.3 * x2 + 0.6 * x3;

            retVal[y * width + x] = scalar < threshold;
        }
    }
    return retVal;
}
