cmake_minimum_required(VERSION 3.22)
project(Project CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)


# nix-shell shell.nix --command "libpng-config --ldflags --cflags" | sed -n -e 's/.*-[LI]\([^ ]*\).*/\1/p'
set(COMMAND_LIGNE_1 "/nix/store/gyh1dxyxxbnq8swn6n4n4if954qhizxy-libpng-apng-1.6.37/lib")
set(COMMAND_LIGNE_2 "/nix/store/naf0ibmgmbmaxssl2nln2qdcj6jpxdb4-libpng-apng-1.6.37-dev/include/libpng16")


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -L${COMMAND_LIGNE_1}")
include_directories("${COMMAND_LIGNE_2}")

add_library(libpng STATIC IMPORTED src/ColorFeaturesExtraction.cpp src/ColorFeaturesExtraction.hh)
set_target_properties(libpng PROPERTIES IMPORTED_LOCATION "${COMMAND_LIGNE_1}/libpng16.so.16.37.0")


add_executable(Project main.cu io.cpp io.h src/ColorSimilarityMeasures.hh src/ColorSimilarityMeasures.cpp src/TextureSimilarityMeasures.cpp src/TextureSimilarityMeasures.hh src/TextureFeaturesExtraction.cpp src/TextureFeaturesExtraction.hh src/Classification.cpp src/Classification.hh)

target_link_libraries(Project libpng)

set_target_properties(Project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
