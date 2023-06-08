//
// Created by Nico on 25/05/2023.
//
#include "ColorSimilarityMeasures.hh"

std::pair<double, double> *ColorSimilarityMeasures(ColorRG *img1, ColorRG *img2, int width, int height) {
    std::pair<double, double> *retVal = new std::pair<double, double> [width * height];


    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            double r = (double) std::min(img1[y * width + x].r, img2[y * width + x].r) /
					(double) std::max(img1[y * width + x].r, img2[y * width + x].r);

			double g = (double) std::min(img1[y * width + x].g, img2[y * width + x].g) /
					(double) std::max(img1[y * width + x].g, img2[y * width + x].g);

            retVal[y * width + x] = std::make_pair(r, g);
        }
    }
    return retVal;
}
