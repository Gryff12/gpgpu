#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include "src/TextureFeaturesExtraction.hh"
#include "src/ColorSimilarityMeasures.hh"
#include "src/TextureSimilarityMeasures.hh"
#include "src/Classification.hh"
#include "io.h"

/*__host__ uint8_t TextureFeaturesExtraction(const thrust::device_vector<uint8_t>& image, int width, int height, int x, int y) {
	uint8_t centerPixel = image[y * width + x];

	uint8_t lbpCode = 0;
	lbpCode |= (image[(y - 1) * width + (x - 1)] >= centerPixel) << 7;
	lbpCode |= (image[(y - 1) * width + x] >= centerPixel) << 6;
	lbpCode |= (image[(y - 1) * width + (x + 1)] >= centerPixel) << 5;
	lbpCode |= (image[y * width + (x + 1)] >= centerPixel) << 4;
	lbpCode |= (image[(y + 1) * width + (x + 1)] >= centerPixel) << 3;
	lbpCode |= (image[(y + 1) * width + x] >= centerPixel) << 2;
	lbpCode |= (image[(y + 1) * width + (x - 1)] >= centerPixel) << 1;
	lbpCode |= (image[y * width + (x - 1)] >= centerPixel);

	return lbpCode;
}*/

int main() {
    unsigned int width, height;


    //This path needs to be edited for each user
    //Must be RGB only :)
    std::string filename_1 = "../dataset/video_frames/0061.png";
    std::string filename_2 = "../dataset/video_frames/0062.png";
    Color **image_1 = loadImage(filename_1, width, height);
    Color **image_2 = loadImage(filename_2, width, height);
    if (image_1) {
        // L'image a été chargée avec succès
        // Faites ce que vous voulez avec l'image ici

        std::cout << "Largeur : " << width << std::endl;
        std::cout << "Hauteur : " << height << std::endl;
        std::cout << "Nombre de pixels : " << width * height << std::endl;
    } else {
        std::cout << "Fail" << std::endl;
    }
    std::cout << std::endl;
    if (image_2) {
        std::cout << "Largeur : " << width << std::endl;
        std::cout << "Hauteur : " << height << std::endl;
        std::cout << "Nombre de pixels : " << width * height << std::endl;
    } else {
        std::cout << "Fail" << std::endl;
    }
    std::cout << std::endl;

    bool **backgroundPixels = isBackgroundPixel(image_1, image_2, width, height, 0.67);

    saveImage("res.ppm", backgroundPixels, width, height);
    return 0;
}
