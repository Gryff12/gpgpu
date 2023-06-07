#include "Classification.hh"
#include "../benchmark/benchmark.h"


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

__global__ void IsBackgroundPixelKernel(char *d_retVal, size_t pitch_ret, double *d_textureSimilarity,
                                        size_t pitch_texture, double *d_colorFeatures_r, size_t pitch_r,
                                        double *d_colorFeatures_g, size_t pitch_g, double threshold, int width,
                                        int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        double x1 = d_textureSimilarity[y * pitch_texture / sizeof(double) + x];
        double x2 = d_colorFeatures_r[y * pitch_r / sizeof(double) + x];
        double x3 = d_colorFeatures_g[y * pitch_g / sizeof(double) + x];
        sortThreeValues(x1, x2, x3);
        double scalar = 0.1 * x1 + 0.3 * x2 + 0.6 * x3;
        char *row = (char *)d_retVal + y * pitch_ret;
        row[x] = (scalar < threshold);
    }
}

bool *IsBackgroundPixel(ColorRG *img1, ColorRG *img2, int width, int height, double threshold) {
    cudaError_t err = cudaSuccess;

    // Allocate memory for the result on the CPU
    bool *retVal = new bool[width * height];

    // Allocate memory for the input images on the GPU
    size_t pitch1;
    size_t pitch2;
    ColorRG *d_img1, *d_img2;

    auto s = width * sizeof(ColorRG);

	err = cudaMallocPitch(&d_img1, &pitch1, s, height);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate memory for d_img1 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMallocPitch(&d_img2, &pitch2, s, height);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate memory for d_img2 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	// Copy input images from CPU to GPU
	err = cudaMemcpy2D(d_img1, pitch1, img1, width * sizeof(ColorRG), width * sizeof(ColorRG), height,
					   cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy img1 from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy2D(d_img2, pitch2, img2, width * sizeof(ColorRG), width * sizeof(ColorRG), height,
					   cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy img2 from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


    // Allocate memory for the texture similarity measures on the GPU
    size_t textureSimilarityPitch;
    double *d_textureSimilarity;
    err = cudaMallocPitch(&d_textureSimilarity, &textureSimilarityPitch, width * sizeof(double), height);
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


    //std::cout << "texture_similarity_kernel_benchmark time: " << texture_similarity_kernel_benchmark.ms << " ms" << std::endl;

    // Allocate memory for the color similarity measures on the GPU
    size_t colorFeaturesPitch_r;
    double *d_colorFeatures_r;
    size_t colorFeaturesPitch_g;
    double *d_colorFeatures_g;
    err = cudaMallocPitch(&d_colorFeatures_r, &colorFeaturesPitch_r, width * sizeof(double), height);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for d_colorFeatures_r (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMallocPitch(&d_colorFeatures_g, &colorFeaturesPitch_g, width * sizeof(double), height);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for d_colorFeatures_g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


	ColorSimilarityKernel<<<gridDim, blockDim>>>(d_img1, pitch1, d_img2, pitch2, d_colorFeatures_r,
											 colorFeaturesPitch_r, d_colorFeatures_g, colorFeaturesPitch_g, width,
											 height);
	cudaDeviceSynchronize();


    //std::cout << "color_similarity_kernel_benchmark time: " << color_similarity_kernel_benchmark.ms << " ms" << std::endl;


    

    // Allocate memory for the result on the GPU
    char *d_retVal;
    size_t pitch;
    err = cudaMallocPitch(&d_retVal, &pitch, width * sizeof(char), height);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for d_retVal (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the IsBackgroundPixelKernel with benchmark
	IsBackgroundPixelKernel<<<gridDim, blockDim>>>(d_retVal, pitch, d_textureSimilarity, textureSimilarityPitch,
												   d_colorFeatures_r, colorFeaturesPitch_r, d_colorFeatures_g,
												   colorFeaturesPitch_g,
												   threshold, width, height); // This line makes it crash
	cudaDeviceSynchronize();

    //std::cout << "is_background_pixel_kernel_benchmark time: " << is_background_pixel_kernel_benchmark.ms << " ms" << std::endl;


    // Copy the result from GPU to CPU
	err = cudaMemcpy2D(retVal, width * sizeof(char), d_retVal, pitch, width * sizeof(char), height, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess && 0) {
        fprintf(stderr, "Failed to copy retVal from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();

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
