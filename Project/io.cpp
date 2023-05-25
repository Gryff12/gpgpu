//
// Created by Maxime on 25/05/2023.
//

#include "io.h"

bool loadImage(const std::string& filename, std::vector<unsigned char>& image, unsigned& width, unsigned& height) {
	FILE* file = fopen(filename.c_str(), "rb");
	if (!file) {
		std::cout << "Erreur lors de l'ouverture du fichier : " << filename << std::endl;
		return false;
	}

	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png) {
		fclose(file);
		std::cout << "Erreur lors de la création de la structure de lecture PNG." << std::endl;
		return false;
	}

	png_infop info = png_create_info_struct(png);
	if (!info) {
		png_destroy_read_struct(&png, NULL, NULL);
		fclose(file);
		std::cout << "Erreur lors de la création de la structure d'informations PNG." << std::endl;
		return false;
	}

	if (setjmp(png_jmpbuf(png))) {
		png_destroy_read_struct(&png, &info, NULL);
		fclose(file);
		std::cout << "Erreur lors de la lecture de l'image PNG." << std::endl;
		return false;
	}

	png_init_io(png, file);
	png_read_info(png, info);

	width = png_get_image_width(png, info);
	height = png_get_image_height(png, info);

	png_byte color_type = png_get_color_type(png, info);
	png_byte bit_depth = png_get_bit_depth(png, info);

	if (bit_depth == 16)
		png_set_strip_16(png);

	if (color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png);

	if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
		png_set_expand_gray_1_2_4_to_8(png);

	if (png_get_valid(png, info, PNG_INFO_tRNS))
		png_set_tRNS_to_alpha(png);

	if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY ||
		color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

	if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
		png_set_gray_to_rgb(png);

	png_read_update_info(png, info);

	png_bytep* row_pointers = new png_bytep[height];
	for (unsigned int y = 0; y < height; y++)
		row_pointers[y] = new png_byte[png_get_rowbytes(png, info)];

	png_read_image(png, row_pointers);

	image.resize(width * height);
	for (unsigned int y = 0; y < height; y++) {
		png_bytep row = row_pointers[y];
		for (unsigned int x = 0; x < width; x++)
			image[y * width + x] = row[x];
	}

	for (unsigned int y = 0; y < height; y++)
		delete[] row_pointers[y];
	delete[] row_pointers;

	png_destroy_read_struct(&png, &info, NULL);
	fclose(file);

	return true;
}