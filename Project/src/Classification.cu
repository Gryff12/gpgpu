#include "Classification.hh"

__device__ void swap(double &a, double &b) {
    double temp = a;
    a = b;
    b = temp;
}

__device__ void sortThreeValues(double &a, double &b, double &c) {
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

__global__ void
IsBackgroundPixelKernel(bool *d_retVal, size_t pitch_ret, double *d_textureSimilarity, size_t pitch_texture,
                        double *d_colorFeatures_r, size_t pitch_r, double *d_colorFeatures_g, size_t pitch_g,
                        double threshold, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        double x1 = d_textureSimilarity[y * pitch_texture + x];
        double x2 = d_colorFeatures_r[y * pitch_r + x];
        double x3 = d_colorFeatures_g[y * pitch_g + x];
        sortThreeValues(x1, x2, x3);
        double scalar = 0.1 * x1 + 0.3 * x2 + 0.6 * x3;
        d_retVal[y * pitch_ret + x] = (scalar < threshold);
    }
}

bool *IsBackgroundPixel(Color *img1, Color *img2, int width, int height, double threshold) {
    cudaError_t err = cudaSuccess;

    // Allocate memory for the result on the CPU
    bool *retVal = new bool[width * height];

    // Allocate memory for the input images on the GPU
    size_t pitch1;
    size_t pitch2;
    Color *d_img1, *d_img2;
    err = cudaMallocPitch((void **) &d_img1, &pitch1, width * sizeof(Color), height);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for d_img1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMallocPitch((void **) &d_img2, &pitch2, width * sizeof(Color), height);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for d_img2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy input images from CPU to GPU
    err = cudaMemcpy2D(d_img1, pitch1, img1, width * sizeof(Color), width * sizeof(Color), height,
                       cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy img1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy2D(d_img2, pitch2, img2, width * sizeof(Color), width * sizeof(Color), height,
                       cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy img2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate memory for the texture similarity measures on the GPU
    size_t textureSimilarityPitch;
    double *d_textureSimilarity;
    err = cudaMallocPitch((void **) &d_textureSimilarity, &textureSimilarityPitch, width * sizeof(double), height);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for d_textureSimilarity (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the TextureSimilarityKernel
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    TextureSimilarityKernel<<<gridDim, blockDim>>>(d_img1, pitch1, d_img2, pitch2, d_textureSimilarity,
                                                   textureSimilarityPitch, width, height);
    cudaDeviceSynchronize();

    // Allocate memory for the color similarity measures on the GPU
    size_t colorFeaturesPitch_r;
    double *d_colorFeatures_r;
    size_t colorFeaturesPitch_g;
    double *d_colorFeatures_g;
    err = cudaMallocPitch((void **) &d_colorFeatures_r, &colorFeaturesPitch_r, width * sizeof(double), height);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for d_colorFeatures_r (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMallocPitch((void **) &d_colorFeatures_g, &colorFeaturesPitch_g, width * sizeof(double), height);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for d_colorFeatures_g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the ColorSimilarityKernel
    ColorSimilarityKernel<<<gridDim, blockDim>>>(d_img1, pitch1, d_img2, pitch2, d_colorFeatures_r,
                                                 colorFeaturesPitch_r, d_colorFeatures_g, colorFeaturesPitch_g, width,
                                                 height);
    cudaDeviceSynchronize();

    // Allocate memory for the result on the GPU
    bool *d_retVal;
    size_t pitch;
    err = cudaMallocPitch((void **) &d_retVal, &pitch, width * sizeof(bool), height);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for d_retVal (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the IsBackgroundPixelKernel
    IsBackgroundPixelKernel<<<gridDim, blockDim>>>(d_retVal, pitch, d_textureSimilarity, textureSimilarityPitch,
                                                   d_colorFeatures_r, colorFeaturesPitch_r, d_colorFeatures_g,
                                                   colorFeaturesPitch_g,
                                                   threshold, width, height);
    cudaDeviceSynchronize();

    // Copy the result from GPU to CPU
    err = cudaMemcpy2D(retVal, width * sizeof(bool), d_retVal, pitch, width * sizeof(bool), height,
                       cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy retVal from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free GPU memory
    err = cudaFree(d_img1);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory for d_img1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_img2);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory for d_img2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_textureSimilarity);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory for d_textureSimilarity (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_colorFeatures_r);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory for d_colorFeatures_r (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_colorFeatures_g);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory for d_colorFeatures_g (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_retVal);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory for d_retVal (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return retVal;
}
