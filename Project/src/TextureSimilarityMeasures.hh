//
// Created by Nico on 29/05/2023.
//

#ifndef PROJECT_TEXTURESIMILARITYMEASURES_HH
#define PROJECT_TEXTURESIMILARITYMEASURES_HH

#include <vector>
#include <bit>
#include <bitset>
#include "../io.h"
#include "TextureFeaturesExtraction.hh"



#ifdef GPU
__device__ uint8_t getPixel(ColorRG *image, int x, int y, int width, int height);

__global__ void TextureSimilarityKernel(ColorRG *d_img1, size_t pitch1, ColorRG *d_img2, size_t pitch2, double *d_similarities, size_t pitch, int width, int height);
#else
double *TextureSimilarityMeasures(ColorRG *img1, ColorRG *img2, int width, int height);
#endif


#endif //PROJECT_TEXTURESIMILARITYMEASURES_HH
