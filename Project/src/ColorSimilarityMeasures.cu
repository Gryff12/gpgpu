#include "ColorSimilarityMeasures.hh"

__global__ void ColorSimilarityKernel(Color *d_img1, Color *d_img2, double *d_colorFeatures_r, size_t pitch_r, double *d_colorFeatures_g, size_t pitch_g, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        double r = static_cast<double>(min(d_img1[y * pitch_r / sizeof(Color) + x].r, d_img2[y * pitch_r / sizeof(Color) + x].r)) /
                   static_cast<double>(max(d_img1[y * pitch_r / sizeof(Color) + x].r, d_img2[y * pitch_r / sizeof(Color) + x].r));
        double g = static_cast<double>(min(d_img1[y * pitch_g / sizeof(Color) + x].g, d_img2[y * pitch_g / sizeof(Color) + x].g)) /
                   static_cast<double>(max(d_img1[y * pitch_g / sizeof(Color) + x].g, d_img2[y * pitch_g / sizeof(Color) + x].g));

//        printf("r: %f, g: %f, x: %d, y: %d\n", r, g, x, y);
        printf("img1r: %d, img1g: %d\n", d_img2[y * pitch_r / sizeof(Color) + x].r, d_img2[y * pitch_g / sizeof(Color) + x].g);
        d_colorFeatures_r[y * pitch_r / sizeof(double) + x] = r;
        d_colorFeatures_g[y * pitch_g / sizeof(double) + x] = g;
    }
}
