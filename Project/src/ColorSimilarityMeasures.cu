#include "ColorSimilarityMeasures.hh"

__global__ void ColorSimilarityKernel(ColorRG *d_img1, size_t pitch1, ColorRG *d_img2, size_t pitch2, double *d_colorFeatures_r, size_t pitch_r,
									  double *d_colorFeatures_g, size_t pitch_g, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        double max_r = max(d_img1[y * pitch1 / sizeof(ColorRG) + x].r, d_img2[y * pitch2 / sizeof(ColorRG) + x].r);
        double r = 1;
        if (max_r != 0) {
            r = static_cast<double>(min(d_img1[y * pitch1 / sizeof(ColorRG) + x].r,
                                        d_img2[y * pitch2 / sizeof(ColorRG) + x].r)) / max_r;
        }
        double max_g = max(d_img1[y * pitch1 / sizeof(ColorRG) + x].g, d_img2[y * pitch2 / sizeof(ColorRG) + x].g);
        double g = 1;
        if (max_g != 0) {
            g = static_cast<double>(min(d_img1[y * pitch1 / sizeof(ColorRG) + x].g,
                                        d_img2[y * pitch2 / sizeof(ColorRG) + x].g)) / max_g;
        }
        d_colorFeatures_r[y * pitch_r / sizeof(double) + x] = r;
        d_colorFeatures_g[y * pitch_g / sizeof(double) + x] = g;
    }
}
