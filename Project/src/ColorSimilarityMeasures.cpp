//
// Created by Nico on 25/05/2023.
//
#include "ColorSimilarityMeasures.hh"

std::pair<double, double> **ColorSimilarityMeasures(Color **img1, Color **img2, int width, int height) {
    std::pair<double, double> **retVal = new std::pair<double, double> *[width];
    for (int i = 0; i < width; i++)
        retVal[i] = new std::pair<double, double>[height];

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            double r = (double) std::min(img1[x][y].r, img2[x][y].r) / (double) std::max(img1[x][y].r, img2[x][y].r);
			double g = (double) std::min(img1[x][y].g, img2[x][y].g) / (double) std::max(img1[x][y].g, img2[x][y].g);
            retVal[x][y] = std::make_pair(r, g);
        }
    }
    return retVal;
}
