#include <stdio.h>
#include <iostream>
#include <glob.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <filesystem>
#include "src/ColorSimilarityMeasures.hh"
#include "src/TextureSimilarityMeasures.hh"
#include "src/Classification.hh"
#include "io.h"
#include "benchmark/benchmark.h"

#ifdef GPU
#	define DEVICE "GPU"
//#	pragma message "Compiling for GPU"
#else
#	define DEVICE "CPU"
//#	pragma message "Compiling for CPU"
#endif

int processImages(const std::string& folderPath, const std::string& outputFolder, double *benchmark_time) {

    std::string searchPattern = folderPath + "/*.png";
    glob_t globResult;
    if (glob(searchPattern.c_str(), GLOB_TILDE, nullptr, &globResult) != 0) {
        std::cerr << "Erreur lors de la recherche des fichiers d'image dans le dossier : " << folderPath << std::endl;
        exit(1);
    }

    if (globResult.gl_pathc == 0) {
        std::cerr << "Aucun fichier d'image trouvé dans le dossier : " << folderPath << std::endl;
        globfree(&globResult);
        exit(1);
    }

    unsigned int width = 0;
    unsigned int height = 0;
    ColorRG* prevImg = nullptr;

	int i = 0;

#ifdef GPU
	size_t pitch1, pitch2, pitch, textureSimilarityPitch, colorFeaturesPitch_r, colorFeaturesPitch_g;
	ColorRG *d_img1, *d_img2;
	double *d_textureSimilarity, *d_colorFeatures_r, *d_colorFeatures_g;
	char *d_retVal;
	bool malloced = false;
#endif

	for (i = 0; i < globResult.gl_pathc; ++i) {
        std::string filename = globResult.gl_pathv[i];

		// Loading new image
		ColorRG* img = loadImage(filename, width, height);

        if (!img) {
            std::cerr << "Échec lors du chargement de l'image : " << filename << std::endl;
            continue;
        }

        if (prevImg) {

			// Traitement
#ifdef GPU
			benchmark::time_output benchmark = benchmark::timeit<bool*>([&]() {
				if (!malloced) {
					malloced = true;
					// Allocate memory for the input images on the GPU
					auto s = width * sizeof(ColorRG);

					auto err = cudaMallocPitch(&d_img1, &pitch1, s, height);
					if (err != cudaSuccess) {
						std::cerr << "Failed to allocate memory for d_img1 (error code " << cudaGetErrorString(err) << ")!" << std::endl;
						exit(EXIT_FAILURE);
					}
					err = cudaMallocPitch(&d_img2, &pitch2, s, height);
					if (err != cudaSuccess) {
						std::cerr << "Failed to allocate memory for d_img2 (error code " << cudaGetErrorString(err) << ")!" << std::endl;
						exit(EXIT_FAILURE);
					}

					// Allocate memory for the texture similarity measures on the GPU
					err = cudaMallocPitch(&d_textureSimilarity, &textureSimilarityPitch, width * sizeof(double), height);
					if (err != cudaSuccess) {
						std::cerr << "Failed to allocate memory for d_textureSimilarity (error code " << cudaGetErrorString(err) << ")!" << std::endl;
						exit(EXIT_FAILURE);
					}

					// Allocate memory for the color similarity measures on the GPU
					err = cudaMallocPitch(&d_colorFeatures_r, &colorFeaturesPitch_r, width * sizeof(double), height);
					if (err != cudaSuccess) {
						std::cerr << "Failed to allocate memory for d_colorFeatures_r (error code " << cudaGetErrorString(err) << ")!" << std::endl;
						exit(EXIT_FAILURE);
					}
					err = cudaMallocPitch(&d_colorFeatures_g, &colorFeaturesPitch_g, width * sizeof(double), height);
					if (err != cudaSuccess) {
						std::cerr << "Failed to allocate memory for d_colorFeatures_g (error code " << cudaGetErrorString(err) << ")!" << std::endl;
						exit(EXIT_FAILURE);
					}

					// Allocate memory for the result on the GPU
					err = cudaMallocPitch(&d_retVal, &pitch, width * sizeof(char), height);
					if (err != cudaSuccess) {
						std::cerr << "Failed to allocate memory for d_retVal (error code " << cudaGetErrorString(err) << ")!" << std::endl;
						exit(EXIT_FAILURE);
					}
				}

				return IsBackgroundPixelGPU(prevImg, img, width, height, 0.67, pitch1, pitch2, d_img1, d_img2, d_textureSimilarity, textureSimilarityPitch, d_colorFeatures_r, colorFeaturesPitch_r, d_colorFeatures_g, colorFeaturesPitch_g, d_retVal, pitch);
            });
#else
			benchmark::time_output benchmark = benchmark::timeit<bool*>([&]() {
				return IsBackgroundPixel(prevImg, img, width, height, 0.67);
			});
#endif

            bool* backgroundPixels = benchmark.result;
			*benchmark_time += benchmark.ms;

			// Enregistrement
            std::string baseName = filename.substr(filename.find_last_of("/") + 1);
            std::string outputFilename = outputFolder + "/" + baseName + ".ppm";
            saveImage(outputFilename.c_str(), backgroundPixels, width, height);


            delete[] backgroundPixels;
            delete[] prevImg;
        }

        prevImg = new ColorRG[width * height];
        std::copy(img, img + width * height, prevImg);

        delete[] img;
    }
    
    // TODO: Free memory
#ifdef GPU
	benchmark::time_output benchmark = benchmark::timeit<void*>([&]() {
		auto err = cudaFree(d_img2);
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
		
		return nullptr;
	});
	
	*benchmark_time += benchmark.ms;

#endif

    if (prevImg) {
        delete[] prevImg;
    }

    globfree(&globResult);
    return i;
}

int main(int argc, char** argv) {
	
	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " inputDirectory outputDirectory" << std::endl;
		std::cout << std::endl;
		return 1;
	}

	std::cout << "Running on " << DEVICE << std::endl;

	// Change this to your own path to the dataset folder and the output folder
	std::string folderPath(argv[1]);
	std::string outputFolder(argv[2]);
	//std::string folderPath = "/home/maxime.madrau/dataset/video_frames";
	//std::string outputFolder = "/home/maxime.madrau/result";

	//std::string folderPath = "/Users/maxime/Documents/Epita/ING 2/Projets/gpgpu/dataset/video_frames";
	//std::string outputFolder = "/Users/maxime/result";

	// Input folder verification
	if (!std::filesystem::exists(folderPath)) {
		std::cerr << "Specified path does not exist" << std::endl;
		return 1;
	}

	// Output folder verification
	if (std::filesystem::exists(outputFolder)) {
		if (!std::filesystem::is_directory(outputFolder)) {
			std::cerr << "Specified path is not a directory" << std::endl;
			return 1;
		}
	} else if (!std::filesystem::create_directory(outputFolder)) {
		std::cerr << "Unable to create the directory" << std::endl;
		return 1;
	}


		double benchmark_time = 0;


	// Init malloc pitch with fake values
#ifdef GPU
	char *dummy_value;
	size_t dummy_pitch;
	cudaMallocPitch(&dummy_value, &dummy_pitch, 1, 1);
#endif


	int no_images = processImages(folderPath, outputFolder, &benchmark_time);

	std::cout << "Images traîtées : " << no_images << std::endl;
	std::cout << "Temps total : " << benchmark_time << " ms" << std::endl;
	std::cout << "fps : " << no_images / (benchmark_time / 1000) << std::endl;

	//std::cout << benchmark.ms << " ms" << std::endl;
	return 0;
}
