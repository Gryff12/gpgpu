//
// Created by Nico on 29/05/2023.
//

#include "TextureFeaturesExtraction.hh"

uint8_t getPixel(ColorRG *image, int x, int y, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0;
    }
    return image[y * width + x].r / 2 + image[y * width + x].g / 2;
}

uint8_t *TextureFeaturesExtraction(ColorRG *image, int width, int height) {
    // Init 2d array of size width, height
    uint8_t *lbpCode = new uint8_t [width * height];

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            //std::cout << "x: " << x << " y: " << y << std::endl;
            uint8_t centerPixel = image[y * width + x].r / 2 + image[y * width + x].g / 2;
            uint8_t current_lbpCode = 0;
            current_lbpCode |= (getPixel(image, x - 1, y - 1, width, height) < centerPixel) << 7;
            current_lbpCode |= (getPixel(image, x, y - 1, width, height) < centerPixel) << 6;
            current_lbpCode |= (getPixel(image, x + 1, y - 1, width, height) < centerPixel) << 5;
            current_lbpCode |= (getPixel(image, x + 1, y, width, height) < centerPixel) << 4;
            current_lbpCode |= (getPixel(image, x + 1, y + 1, width, height) < centerPixel) << 3;
            current_lbpCode |= (getPixel(image, x, y + 1, width, height) < centerPixel) << 2;
            current_lbpCode |= (getPixel(image, x - 1, y + 1, width, height) < centerPixel) << 1;
            current_lbpCode |= (getPixel(image, x - 1, y, width, height) < centerPixel);

            lbpCode[y * width + x] = current_lbpCode;
        }
    }
    return lbpCode;
}
