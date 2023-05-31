//
// Created by Nico on 29/05/2023.
//

#ifndef PROJECT_TEXTURESIMILARITYMEASURES_HH
#define PROJECT_TEXTURESIMILARITYMEASURES_HH

#include <vector>
#include <bit>
#include <bitset>
#include <cstdint>
#include "../io.h"

double **TextureSimilarityMeasures(Color **img1, Color **img2, int width, int height);

__device__ uint8_t getPixel(Color *image, int x, int y, size_t pitch, int width, int height);

__global__ void TextureSimilarityKernel(Color *d_img1, Color *d_img2, double *d_similarities, size_t pitch, int width, int height);

#endif //PROJECT_TEXTURESIMILARITYMEASURES_HH
