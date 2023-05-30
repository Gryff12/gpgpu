//
// Created by Nico on 29/05/2023.
//

#include "Classification.hh"
#include <png.h>

template <typename T>
void swap(T& a, T& b) {
    float temp = a;
    a = b;
    b = temp;
}

template <typename T>
void sortThreeValues(T& a, T& b, T& c) {
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

void saveImage(const char *filename, double** pixels, int width, int height) {
	FILE *f = fopen(filename, "w");
	(void) fprintf(f, "P6\n%d %d\n255\n", width, height);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; ++x) {
			static unsigned char color[3];
			color[0] = (unsigned char) (std::min(1., pixels[x][y]) * 255);
			color[1] = (unsigned char) (std::min(1., pixels[x][y]) * 255);
			color[2] = (unsigned char) (std::min(1., pixels[x][y]) * 255);

			(void) fwrite(color, 1, 3, f);
		}
	}
	fclose(f);
}

double** classifactionIndicators(int width, int height, uint8_t** lbpCode, double** colorSimilarityMeasures, double** textureSimilarityMeasures) {

	double** classificationIndicators = new double*[width];
	for (int i = 0; i < width; i++)
		classificationIndicators[i] = new double[height];

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			double a = static_cast<double>(lbpCode[i][j]) / 255.;
			double b = colorSimilarityMeasures[i][j];
			double c = textureSimilarityMeasures[i][j];

			//Order elements in ascending order
			sortThreeValues(a, b, c);
			classificationIndicators[i][j] = a * 0.1 + b * 0.3 + c * 0.6;

		}
	}

    return classificationIndicators;
}