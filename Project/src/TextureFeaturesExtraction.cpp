//
// Created by Nico on 29/05/2023.
//

#include "TextureFeaturesExtraction.hh"


Color getPixel(Color** image, int x, int y, int width, int height) {
	if (x < 0 || x >= width || y < 0 || y >= height) {
		return Color(0, 0, 0);
	}
	return image[x][y];
}

uint8_t** TextureFeaturesExtraction(Color** image, int width, int height) {
	// Init 2d array of size width, height
	uint8_t ** lbpCode = new uint8_t*[width];
	for (int i = 0; i < width; i++)
		lbpCode[i] = new uint8_t[height];

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			//std::cout << "x: " << x << " y: " << y << std::endl;
			double centerPixel = image[x][y].r;

			uint8_t current_lbpCode = 0;
            current_lbpCode |= (getPixel(image, x - 1, y - 1, width, height).r >= centerPixel) << 7;
            current_lbpCode |= (getPixel(image, x, y - 1, width, height).r >= centerPixel) << 6;
            current_lbpCode |= (getPixel(image, x + 1, y - 1, width, height).r >= centerPixel) << 5;
            current_lbpCode |= (getPixel(image, x + 1, y, width, height).r >= centerPixel) << 4;
            current_lbpCode |= (getPixel(image, x + 1, y + 1, width, height).r >= centerPixel) << 3;
            current_lbpCode |= (getPixel(image, x, y + 1, width, height).r >= centerPixel) << 2;
            current_lbpCode |= (getPixel(image, x - 1, y + 1, width, height).r >= centerPixel) << 1;
            current_lbpCode |= (getPixel(image, x - 1, y, width, height).r >= centerPixel);

			lbpCode[x][y] = current_lbpCode;
        }
    }

    return lbpCode;
}