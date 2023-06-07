#include <stdio.h>
#include <iostream>
#include <glob.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include <thread>
#include "src/ColorSimilarityMeasures.hh"
#include "src/TextureSimilarityMeasures.hh"
#include "src/Classification.hh"
#include "io.h"

#include "benchmark/benchmark.h"

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
            benchmark::time_output benchmark = benchmark::timeit<bool*>([prevImg, img, width, height]() {
                return IsBackgroundPixel(prevImg, img, width, height, 0.67);
            });

            bool* backgroundPixels = benchmark.result;
			*benchmark_time += benchmark.ms;

			// Enregistrement
            std::string baseName = filename.substr(filename.find_last_of("/") + 1);
            std::string outputFilename = outputFolder + "/" + baseName + ".ppm";
            //std::cout << "Enregistrement de l'image : " << outputFilename << std::endl;
            saveImage(outputFilename.c_str(), backgroundPixels, width, height);


            delete[] backgroundPixels;
            delete[] prevImg;
        }

        prevImg = new ColorRG[width * height];
        std::copy(img, img + width * height, prevImg);

        delete[] img;
    }

    if (prevImg) {
        delete[] prevImg;
    }

    globfree(&globResult);
    return i;
}

int main() {
    // Change this to your own path to the dataset folder and the output folder
    std::string folderPath = "/home/maxime.madrau/dataset/video_frames";
    std::string outputFolder = "/home/maxime.madrau/result";

	double benchmark_time = 0;


	// Init malloc pitch with fake values
	char* dummy_value;
	size_t dummy_pitch;
	cudaMallocPitch(&dummy_value, &dummy_pitch, 1, 1);


	int no_images = processImages(folderPath, outputFolder, &benchmark_time);

	std::cout << "Images traîtées : " << no_images << std::endl;
    std::cout << "Temps total : " << benchmark_time << " ms" << std::endl;
	std::cout << "fps : " << no_images / (benchmark_time / 1000) << std::endl;
    return 0;
}
