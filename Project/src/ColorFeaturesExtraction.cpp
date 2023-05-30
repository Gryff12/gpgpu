//
// Created by Nico on 25/05/2023.
//

#include "ColorFeaturesExtraction.hh"

// ColorFeaturesExtraction is a function that takes an image and returns a vector of the sum of the red and green components of each pixel
// Perhaps, we should remove the for loops.
uint8_t** ColorFeaturesExtraction(Color** image, unsigned int width, unsigned int height){
	// Init 2d array of size width, height
	uint8_t** redGreen = new uint8_t*[width];
	for (int i = 0; i < width; i++)
		redGreen[i] = new uint8_t[height];

    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
			uint8_t red = image[x][y].r;
			uint8_t green = image[x][y].g;

			uint8_t sum = red + green;
            redGreen[x][y] = sum;
        }
    }
    return redGreen;
}