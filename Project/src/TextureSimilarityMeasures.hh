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

double *TextureSimilarityMeasures(Color *img1, Color *img2, int width, int height);



__device__ uint8_t getPixel(Color *image, int x, int y, int width, int height);

__global__ void TextureSimilarityKernel(Color *d_img1, size_t pitch1, Color *d_img2, size_t pitch2, double *d_similarities, size_t pitch, int width, int height);


#endif //PROJECT_TEXTURESIMILARITYMEASURES_HH
