#include "Classification.hh"

void swap(double &a, double &b) {
    double temp = a;
    a = b;
    b = temp;
}

void sortThreeValues(double &a, double &b, double &c) {
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

__global__ void IsBackgroundPixelKernel(bool *d_retVal, double *d_textureSimilarity, std::pair<double, double> *d_colorFeatures, double threshold, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;

        double x1 = d_textureSimilarity[index];
        double x2 = d_colorFeatures[index].first;
        double x3 = d_colorFeatures[index].second;

        sortThreeValues(x1, x2, x3);

        double scalar = 0.1 * x1 + 0.3 * x2 + 0.6 * x3;

        d_retVal[index] = (scalar < threshold);
    }
}

bool **IsBackgroundPixel(Color **img1, Color **img2, int width, int height, double threshold) {
    // Allocate memory for the result on the CPU
    bool **retVal = new bool *[width];
    for (int i = 0; i < width; i++)
        retVal[i] = new bool[height];

    // Allocate memory for the input images on the GPU
    size_t pitch;
    Color *d_img1, *d_img2;
    cudaMallocPitch((void **)&d_img1, &pitch, width * sizeof(Color), height);
    cudaMallocPitch((void **)&d_img2, &pitch, width * sizeof(Color), height);

    // Copy input images from CPU to GPU
    cudaMemcpy2D(d_img1, pitch, img1[0], width * sizeof(Color), width * sizeof(Color), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_img2, pitch, img2[0], width * sizeof(Color), width * sizeof(Color), height, cudaMemcpyHostToDevice);

    // Allocate memory for the texture similarity measures on the GPU
    size_t textureSimilarityPitch;
    double *d_textureSimilarity;
    cudaMallocPitch((void **)&d_textureSimilarity, &textureSimilarityPitch, width * sizeof(double), height);

    // Launch the TextureSimilarityKernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    TextureSimilarityKernel<<<gridDim, blockDim>>>(d_img1, d_img2, d_textureSimilarity, textureSimilarityPitch, width, height);

    // Allocate memory for the color similarity measures on the GPU
    size_t colorFeaturesPitch;
    std::pair<double, double> *d_colorFeatures;
    cudaMallocPitch((void **)&d_colorFeatures, &colorFeaturesPitch, width * sizeof(std::pair<double, double>), height);

    // Launch the ColorSimilarityKernel
    ColorSimilarityKernel<<<gridDim, blockDim>>>(d_img1, d_img2, d_colorFeatures, colorFeaturesPitch, width, height);

    // Allocate memory for the result on the GPU
    bool *d_retVal;
    cudaMallocPitch((void **)&d_retVal, &pitch, width * sizeof(bool), height);

    // Launch the IsBackgroundPixelKernel
    IsBackgroundPixelKernel<<<gridDim, blockDim>>>(d_retVal, d_textureSimilarity, d_colorFeatures, threshold, width, height);

    // Copy the result from GPU to CPU
    cudaMemcpy2D(retVal[0], width * sizeof(bool), d_retVal, pitch, width * sizeof(bool), height, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_textureSimilarity);
    cudaFree(d_colorFeatures);
    cudaFree(d_retVal);

    return retVal;
}
