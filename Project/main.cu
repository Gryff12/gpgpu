#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include "src/ColorSimilarityMeasures.hh"
#include "src/TextureSimilarityMeasures.hh"
#include "src/Classification.hh"
#include "io.h"

#include "benchmark/benchmark.h"

int main() {

    unsigned int width, height;

    benchmark::time_output load_image_1_benchmark = benchmark::timeit<Color*>([&width, &height](){
        std::string filename_1 = "/home/maxime.madrau/dataset/video_frames/0061.png";
        Color **image_1 = loadImage(filename_1, width, height);
        if (image_1) {
            // L'image a été chargée avec succès
            // Faites ce que vous voulez avec l'image ici

            std::cout << "Largeur : " << width << std::endl;
            // 640
            std::cout << "Hauteur : " << height << std::endl;
            // 360
            std::cout << "Nombre de pixels : " << width * height << std::endl;
        } else {
            std::cout << "Fail" << std::endl;
        }

        Color *img_1 = new Color[width * height];
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; ++y)
                img_1[y * width + x] = image_1[x][y];

        return img_1;
    });

    Color *img_1 = load_image_1_benchmark.result;

    std::cout << "load_image_1_benchmark: " << load_image_1_benchmark.ms << " ms" << std::endl;

    benchmark::time_output load_image_2_benchmark = benchmark::timeit<Color*>([&width, &height](){
        std::string filename_2 = "/home/maxime.madrau/dataset/video_frames/0062.png";
        Color **image_2 = loadImage(filename_2, width, height);
        if (image_2) {
            std::cout << "Largeur : " << width << std::endl;
            std::cout << "Hauteur : " << height << std::endl;
            std::cout << "Nombre de pixels : " << width * height << std::endl;
        } else {
            std::cout << "Fail" << std::endl;
        }

        Color *img_2 = new Color[width * height];
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; ++y)
                img_2[y * width + x] = image_2[x][y];

        return img_2;
    });

    Color *img_2 = load_image_2_benchmark.result;

    std::cout << "load_image_2_benchmark: " << load_image_2_benchmark.ms << " ms" << std::endl;



    //benchmark::time_output total_benchmark = benchmark::timeit<void*>([](){

    
    benchmark::time_output benchmark = benchmark::timeit<bool*>([img_1, img_2, width, height]() {
		return IsBackgroundPixel(img_1, img_2, width, height, 0.67);
	});

	bool* backgroundPixels = benchmark.result;

	std::cout << "Benchmark (traitement): " << benchmark.ms << " ms" << std::endl;

    //bool *backgroundPixels = IsBackgroundPixel(img_1, img_2, width, height, 0.67);

    saveImage("/afs/cri.epita.fr/user/m/ma/maxime.madrau/u/resr.ppm", backgroundPixels, width, height);

    //delete [] img_1;
    //delete [] img_2;
    //delete [] image_1;
    //delete [] image_2;
    //delete [] backgroundPixels;
    
//    return nullptr;
//
//    });
//
//	std::cout << "Benchmark (total): " << total_benchmark.ms << " ms" << std::endl;

    return 0;
}
