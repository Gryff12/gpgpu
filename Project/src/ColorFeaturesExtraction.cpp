//
// Created by Nico on 25/05/2023.
//

#include "ColorFeaturesExtraction.hh"

// ColorFeaturesExtraction is a function that takes an image and returns a vector of the sum of the red and green components of each pixel
// Perhaps, we should remove the for loops.
std::vector<unsigned int> ColorFeaturesExtraction(std::vector<unsigned char> image, unsigned int width, unsigned int height){
    std::vector<unsigned int> redGreen; //Sould be equals to image.size/4

    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned int index = (y * width + x) * 3; // Chaque pixel a 4 composantes (RGBA)
            unsigned char red = image[index];
            unsigned char green = image[index + 1];

            unsigned int sum = red + green;
            redGreen.push_back(sum);
        }
    }
    return redGreen;
}