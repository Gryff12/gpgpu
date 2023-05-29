//
// Created by Nico on 29/05/2023.
//

#include "Classification.hh"
#include <png.h>

void swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
}

void sortThreeValues(float& a, float& b, float& c) {
    if (a > b) {
        swap(a, b);
    }

    if (a > c) {
        swap(a, c);
    }

    if (b > c) {
        swap(b, c);
    }
}

void saveImage(const char *filename, std::vector<float> pixels, int width, int height) {
	FILE *f = fopen(filename, "w");
	(void) fprintf(f, "P6\n%d %d\n255\n", width, height);
	for (int i = 0; i < width * height; ++i) {
		static unsigned char color[3];
		color[0] = (unsigned char) (std::min(1.f, pixels[i]) * 255);
		color[1] = (unsigned char) (std::min(1.f, pixels[i]) * 255);
		color[2] = (unsigned char) (std::min(1.f, pixels[i]) * 255);

		(void) fwrite(color, 1, 3, f);
	}
	fclose(f);
}

std::vector<float> classifactionIndicators(std::vector<uint8_t> lbpCode, std::vector<float> colorSimilarityMeasures, std::vector<float> textureSimilarityMeasures){
    std::vector<float> classificationIndicators;
    for (int i = 0; i < lbpCode.size(); i++){
        float a = static_cast<float>(lbpCode[i]) / 255.f;
        float b = colorSimilarityMeasures[i];
        float c = textureSimilarityMeasures[i];

        //Order elements in ascending order

		sortThreeValues(a, b, c);

		if (b < 0) {
			printf("%f %f %f\n", a, b, c);
			printf("%f %f %f\n------\n", a, b, c);
		}
		float value = a * 0.1 + b * 0.3 + c * 0.6;
        classificationIndicators.push_back(value < 0.67 ? 0 : 1);
    }
    return classificationIndicators;
}



