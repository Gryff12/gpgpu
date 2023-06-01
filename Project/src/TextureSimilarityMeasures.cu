#include "TextureSimilarityMeasures.hh"

__device__ uint8_t getPixel(Color *image, int x, int y, size_t pitch, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0;
    }
    Color pixel = image[y * pitch / sizeof(Color) + x];
    return pixel.r / 2 + pixel.g / 2;
}

__global__ void TextureSimilarityKernel(Color *d_img1, size_t pitch1,  Color *d_img2, size_t pitch2, double *d_similarities, size_t pitch, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint8_t centerPixel1 = (d_img1[y * pitch1 / sizeof(Color) + x].r + d_img1[y * pitch1 / sizeof(Color) + x].g) / 2;
        uint8_t centerPixel2 = (d_img2[y * pitch2 / sizeof(Color) + x].r + d_img2[y * pitch2 / sizeof(Color) + x].g) / 2;

        uint8_t current_lbpCode1 = 0;
        current_lbpCode1 |= (getPixel(d_img1, x - 1, y - 1, pitch1, width, height) < centerPixel1) << 7;
        current_lbpCode1 |= (getPixel(d_img1, x, y - 1, pitch1, width, height) < centerPixel1) << 6;
        current_lbpCode1 |= (getPixel(d_img1, x + 1, y - 1, pitch1, width, height) < centerPixel1) << 5;
        current_lbpCode1 |= (getPixel(d_img1, x + 1, y, pitch1, width, height) < centerPixel1) << 4;
        current_lbpCode1 |= (getPixel(d_img1, x + 1, y + 1, pitch1, width, height) < centerPixel1) << 3;
        current_lbpCode1 |= (getPixel(d_img1, x, y + 1, pitch1, width, height) < centerPixel1) << 2;
        current_lbpCode1 |= (getPixel(d_img1, x - 1, y + 1, pitch1, width, height) < centerPixel1) << 1;
        current_lbpCode1 |= (getPixel(d_img1, x - 1, y, pitch1, width, height) < centerPixel1);

        uint8_t current_lbpCode2 = 0;
        current_lbpCode2 |= (getPixel(d_img2, x - 1, y - 1, pitch2, width, height) < centerPixel2) << 7;
        current_lbpCode2 |= (getPixel(d_img2, x, y - 1, pitch2, width, height) < centerPixel2) << 6;
        current_lbpCode2 |= (getPixel(d_img2, x + 1, y - 1, pitch2, width, height) < centerPixel2) << 5;
        current_lbpCode2 |= (getPixel(d_img2, x + 1, y, pitch2, width, height) < centerPixel2) << 4;
        current_lbpCode2 |= (getPixel(d_img2, x + 1, y + 1, pitch2, width, height) < centerPixel2) << 3;
        current_lbpCode2 |= (getPixel(d_img2, x, y + 1, pitch2, width, height) < centerPixel2) << 2;
        current_lbpCode2 |= (getPixel(d_img2, x - 1, y + 1, pitch2, width, height) < centerPixel2) << 1;
        current_lbpCode2 |= (getPixel(d_img2, x - 1, y, pitch2, width, height) < centerPixel2);

         __syncthreads();

        uint8_t diff = current_lbpCode1 ^ current_lbpCode2;
        double count = __popc(diff);

        d_similarities[y * pitch / sizeof(double) + x] = count / 8.0;
    }
}
