#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include "src/ColorFeaturesExtraction.hh"
#include "src/TextureFeaturesExtraction.hh"
#include "src/ColorSimilarityMeasures.hh"
#include "src/TextureSimilarityMeasures.hh"
#include "src/Classification.hh"
#include "test.h"
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
    std::string filename_1 = "/home/maxime.madrau/dataset/video_frames/0060.png";
    std::string filename_2 = "/home/maxime.madrau/dataset/video_frames/0061.png";
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
        // L'image a été chargée avec succès
        // Faites ce que vous voulez avec l'image ici

        std::cout << "Largeur : " << width << std::endl;
        std::cout << "Hauteur : " << height << std::endl;
        std::cout << "Nombre de pixels : " << width * height << std::endl;
    } else {
        std::cout << "Fail" << std::endl;
    }
    std::cout << std::endl;
    int size = width * height;
    //auto redGreen = ColorFeaturesExtraction(image_1, width, height);
    auto lbpCode = TextureFeaturesExtraction(image_1, width, height);
    auto colorSimilarityMeasures = ColorSimilarityMeasures(image_1, image_2, width, height);
    auto textureSimilarityMeasures = TextureSimilarityMeasures(image_1, image_2, width, height);
    auto classification = classifactionIndicators(width, height, lbpCode, colorSimilarityMeasures,
                                                  textureSimilarityMeasures);
//
//    //std::cout << "classification size : " << classification.size() << std::endl;
//    //std::cout << "textureSimilarityMeasures size : " << textureSimilarityMeasures.size() << std::endl;
//    //std::cout << "colorSimilarityMeasures size : " << colorSimilarityMeasures.size() << std::endl;
//    //std::cout << "lbpCode size : " << lbpCode.size() << std::endl;
//    //std::cout << "redGreen size : " << redGreen.size() << std::endl;
//    //std::cout << "size : " << size << std::endl;
//    //std::cout << "redGreen size must be equal to size : " << (redGreen.size() == size) << std::endl;
//
//
    saveImage("/home/maxime.madrau/afs/res.ppm", classification, width, height);
    return 0;
}
