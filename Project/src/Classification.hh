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



#ifdef GPU
__device__ void sortThreeValues(double &a, double &b, double &c);

__device__ void swap(double &a, double &b);

bool *IsBackgroundPixelGPU(ColorRG *img1, ColorRG *img2, int width, int height, double threshold, size_t pitch1, size_t pitch2, ColorRG *d_img1, ColorRG *d_img2, double *d_textureSimilarity, size_t textureSimilarityPitch, double *d_colorFeatures_r, size_t colorFeaturesPitch_r, double *d_colorFeatures_g, size_t colorFeaturesPitch_g, char *d_retVal, size_t pitch);

__global__ void IsBackgroundPixelKernel(char *d_retVal, size_t pitch_ret, double *d_textureSimilarity, size_t pitch_texture,
                        double *d_colorFeatures_r, size_t pitch_r, double *d_colorFeatures_g, size_t pitch_g,
                        double threshold, int width, int height);
#endif
bool *IsBackgroundPixel(ColorRG *img1, ColorRG *img2, int width, int height, double threshold);
double CalculateChoquetIntegral(double x1, double x2, double x3);

#endif //PROJECT_CLASSIFICATION_HH
