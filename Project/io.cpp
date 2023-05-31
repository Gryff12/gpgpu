//
// Created by Maxime on 25/05/2023.
//

#include "io.h"

Color **loadImage(const std::string &filename, unsigned &width, unsigned &height) {
    FILE *file = fopen(filename.c_str(), "rb");
    if (!file) {
        std::cerr << "Erreur lors de l'ouverture du fichier : " << filename << std::endl;
        return nullptr;
    }
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(file);
        std::cerr << "Erreur lors de l'initialisation de la structure de lecture PNG." << std::endl;
        return nullptr;
    }
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(file);
        std::cerr << "Erreur lors de la création de la structure d'information PNG." << std::endl;
        return nullptr;
    }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(file);
        std::cerr << "Erreur lors de la lecture de l'image PNG." << std::endl;
        return nullptr;
    }
    png_init_io(png, file);
    png_read_info(png, info);
    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    png_set_strip_16(png);
    png_set_packing(png);
    int color_type = png_get_color_type(png, info);
    int bit_depth = png_get_bit_depth(png, info);
    if (color_type != PNG_COLOR_TYPE_RGB && color_type != PNG_COLOR_TYPE_RGBA) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(file);
        std::cerr << "Le format de couleur de l'image n'est pas pris en charge." << std::endl;
        return nullptr;
    }
    if (bit_depth != 8) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(file);
        std::cerr << "La profondeur de bits de l'image n'est pas prise en charge." << std::endl;
        return nullptr;
    }
    png_read_update_info(png, info);
    png_bytep *row_pointers = new png_bytep[height];
    for (unsigned int y = 0; y < height; ++y) {
        row_pointers[y] = new png_byte[png_get_rowbytes(png, info)];
    }
    png_read_image(png, row_pointers);
    Color **image = new Color *[width];
    for (int i = 0; i < width; i++)
        image[i] = new Color[height];
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            uint8_t r = row_pointers[y][x * 3];
            uint8_t g = row_pointers[y][x * 3 + 1];
            uint8_t b = row_pointers[y][x * 3 + 2];
            image[x][y] = {r, g, b};
        }
    }
    png_destroy_read_struct(&png, &info, NULL);
    fclose(file);
    for (unsigned int y = 0; y < height; ++y) {
        delete[] row_pointers[y];
    }
    delete[] row_pointers;
    return image;
}

void saveImage(const char *filename, double **pixels, int width, int height) {
    FILE *f = fopen(filename, "w");
    (void) fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; ++x) {
            static unsigned char color[3];
            color[0] = (unsigned char) (std::min(1., pixels[x][y]) * 255);
            color[1] = (unsigned char) (std::min(1., pixels[x][y]) * 255);
            color[2] = (unsigned char) (std::min(1., pixels[x][y]) * 255);
            (void) fwrite(color, 1, 3, f);
        }
    }
    fclose(f);
}

void saveImage(const char *filename, bool *pixels, int width, int height) {
    FILE *f = fopen(filename, "w");
    (void) fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; ++x) {
            static unsigned char color[3];
            color[0] = (unsigned char) pixels[y * width + x] * 255;
            color[1] = (unsigned char) pixels[y * width + x] * 255;
            color[2] = (unsigned char) pixels[y * width + x] * 255;
            (void) fwrite(color, 1, 3, f);
        }
    }
    fclose(f);
}


void saveImage(const char *filename, Color *pixels, int width, int height){
    FILE *f = fopen(filename, "w");
    (void) fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; ++x) {
            static unsigned char color[3];
            color[0] = (unsigned char) pixels[y * width + x].r;
            color[1] = (unsigned char) pixels[y * width + x].g;
            color[2] = (unsigned char) pixels[y * width + x].b;
            (void) fwrite(color, 1, 3, f);
        }
    }
    fclose(f);
}
