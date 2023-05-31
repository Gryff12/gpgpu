#include "ColorSimilarityMeasures.hh"

__global__ void ColorSimilarityKernel(Color *d_img1, Color *d_img2, std::pair<double, double> *d_colorFeatures, size_t pitch, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        double r = static_cast<double>(min(d_img1[y * pitch / sizeof(Color) + x].r, d_img2[y * pitch / sizeof(Color) + x].r)) /
                   static_cast<double>(max(d_img1[y * pitch / sizeof(Color) + x].r, d_img2[y * pitch / sizeof(Color) + x].r));
        double g = static_cast<double>(min(d_img1[y * pitch / sizeof(Color) + x].g, d_img2[y * pitch / sizeof(Color) + x].g)) /
                   static_cast<double>(max(d_img1[y * pitch / sizeof(Color) + x].g, d_img2[y * pitch / sizeof(Color) + x].g));

        d_colorFeatures[y * pitch / sizeof(std::pair<double, double>) + x] = make_pair(r, g);
    }
}
