//
// Created by Nico on 25/05/2023.
//
#include "ColorSimilarityMeasures.hh"

std::pair<double> **ColorSimilarityMeasures(Color **img1, Color **img2, int width, int height) {
    std::pair<double> **retVal = new std::pair<float> *[width];
    for (int i = 0; i < width; i++)
        retVal[i] = new std::pair<float>[height];
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            double r = std::min(img1[x][y].r, img2[x][y].r) / std::max(img1[x][y].r, img2[x][y].r);
            double g = std::min(img1[x][y].g, img2[x][y].g) / std::max(img1[x][y].g, img2[x][y].g);
            retVal[x][y] = std::make_pair(r, g);
        }
    }
    return retVal;
}

}
