//
// Created by Nico on 29/05/2023.
//

#include "TextureFeaturesExtraction.hh"


std::vector<uint8_t> TextureFeaturesExtraction(const std::vector<uint8_t>& image, int width, int height) {
    std::vector<uint8_t> lbpCode;
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++) {
            uint8_t centerPixel = image[y * width + x];

            uint8_t current_lbpCode = 0;
            current_lbpCode |= (image[(y - 1) * width + (x - 1)] >= centerPixel) << 7;
            current_lbpCode |= (image[(y - 1) * width + x] >= centerPixel) << 6;
            current_lbpCode |= (image[(y - 1) * width + (x + 1)] >= centerPixel) << 5;
            current_lbpCode |= (image[y * width + (x + 1)] >= centerPixel) << 4;
            current_lbpCode |= (image[(y + 1) * width + (x + 1)] >= centerPixel) << 3;
            current_lbpCode |= (image[(y + 1) * width + x] >= centerPixel) << 2;
            current_lbpCode |= (image[(y + 1) * width + (x - 1)] >= centerPixel) << 1;
            current_lbpCode |= (image[y * width + (x - 1)] >= centerPixel);

            lbpCode.push_back(current_lbpCode);
        }
    }

    return lbpCode;
}