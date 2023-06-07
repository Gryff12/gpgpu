//
// Created by Nico on 29/05/2023.
//

#ifndef PROJECT_CLASSIFICATION_HH
#define PROJECT_CLASSIFICATION_HH

#include "../io.h"
#include "TextureFeaturesExtraction.hh"
#include "ColorSimilarityMeasures.hh"
#include "TextureSimilarityMeasures.hh"
#include <png.h>
#include <vector>

double CalculateChoquetIntegral(double x1, double x2, double x3);

bool *IsBackgroundPixel(ColorRG *img1, ColorRG *img2, int width, int height, double threshold);


//#ifdef __CUDA_ARCH__

__device__ void sortThreeValues(double &a, double &b, double &c);

__device__ void swap(double &a, double &b);

__global__ void
IsBackgroundPixelKernel(char *d_retVal, size_t pitch_ret, double *d_textureSimilarity, size_t pitch_texture,
                        double *d_colorFeatures_r, size_t pitch_r, double *d_colorFeatures_g, size_t pitch_g,
                        double threshold, int width, int height);
//#endif

#endif //PROJECT_CLASSIFICATION_HH
