#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include "src/ColorFeaturesExtraction.hh"
#include "test.h"
#include "io.h"


__host__ uint8_t TextureFeaturesExtraction(const thrust::device_vector<uint8_t>& image, int width, int height, int x, int y) {
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
}

int main() {

	unsigned int width, height;
	std::vector<unsigned char> image;

    //This path needs to be edited for each user
    //Must be RGB only :)
	std::string filename = "/home/nicolas.muller/afs/cuda/GPGPU 2023-04/dataset/frames/10.png";

	if (loadImage(filename, image, width, height)) {
		// L'image a été chargée avec succès
		// Faites ce que vous voulez avec l'image ici

		std::cout << "Largeur : " << width << std::endl;
		std::cout << "Hauteur : " << height << std::endl;
		std::cout << "Nombre de pixels : " << image.size() << std::endl;
	} else {
		std::cout << "Fail" << std::endl;
	}
    int size = width * height;
    std::vector<unsigned int> redGreen = ColorFeaturesExtraction(image, width, height);
    std::cout << "redGreen size : " << redGreen.size() << std::endl;
    std::cout << "size : " << size << std::endl;
    std::cout << "redGreen size must be equal to size : " << (redGreen.size() == size) << std::endl;

    return 0;
}