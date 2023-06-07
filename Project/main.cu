#include <stdio.h>
#include <iostream>
#include <glob.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include "src/ColorSimilarityMeasures.hh"
#include "src/TextureSimilarityMeasures.hh"
#include "src/Classification.hh"
#include "io.h"

#include "benchmark/benchmark.h"

int processImages(const std::string& folderPath, const std::string& outputFolder) {
    std::string searchPattern = folderPath + "/*.png";
    glob_t globResult;
    if (glob(searchPattern.c_str(), GLOB_TILDE, nullptr, &globResult) != 0) {
        std::cout << "Erreur lors de la recherche des fichiers d'image dans le dossier : " << folderPath << std::endl;
        return 1;
    }

    if (globResult.gl_pathc == 0) {
        std::cout << "Aucun fichier d'image trouvé dans le dossier : " << folderPath << std::endl;
        globfree(&globResult);
        return 1;
    }

    unsigned int width = 0;
    unsigned int height = 0;
    Color* prevImg = nullptr;

    for (size_t i = 0; i < globResult.gl_pathc; ++i) {
        std::string filename = globResult.gl_pathv[i];
        std::cout << "Traitement de l'image : " << filename << std::endl;

        benchmark::time_output loadImageBenchmark = benchmark::timeit<Color*>([&width, &height, filename]() {
            Color** image = loadImage(filename, width, height);
            if (!image) {
                return static_cast<Color*>(nullptr);
            }

            Color* img = new Color[width * height];
            for (unsigned int x = 0; x < width; x++) {
                for (unsigned int y = 0; y < height; y++) {
                    img[y * width + x] = image[x][y];
                }
            }

            //deleteImage(image, width);
            return img;
        });

        Color* img = loadImageBenchmark.result;
        if (!img) {
            std::cout << "Échec lors du chargement de l'image : " << filename << std::endl;
            continue;
        }

        if (prevImg) {
            benchmark::time_output benchmark = benchmark::timeit<bool*>([prevImg, img, width, height]() {
                return IsBackgroundPixel(prevImg, img, width, height, 0.67);
            });

            bool* backgroundPixels = benchmark.result;
            //std::cout << "Temps de traitement : " << benchmark.ms << " ms" << std::endl;

            std::string baseName = filename.substr(filename.find_last_of("/") + 1);
            std::string outputFilename = outputFolder + "/" + baseName + ".ppm";
            std::cout << "Enregistrement de l'image : " << outputFilename << std::endl;
            saveImage(outputFilename.c_str(), backgroundPixels, width, height);

            delete[] backgroundPixels;
        }

        if (prevImg) {
            delete[] prevImg;
        }

        prevImg = new Color[width * height];
        std::copy(img, img + width * height, prevImg);

        delete[] img;
    }

    if (prevImg) {
        delete[] prevImg;
    }

    globfree(&globResult);
    return 0;
}

int main() {
    // Change this to your own path to the dataset folder and the output folder
    std::string folderPath = "/home/nicolas.muller/dataset/video_frames";
    std::string outputFolder = "/home/nicolas.muller/result";

    // Benchmark the whole process
    benchmark::time_output globalTime = benchmark::timeit<int>([folderPath, outputFolder]() {
        return processImages(folderPath, outputFolder);
    });
    std::cout << "Temps total : " << globalTime.ms << " ms" << std::endl;
    return 0;
}
