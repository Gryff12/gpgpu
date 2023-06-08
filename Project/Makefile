CC=g++
NVCC=nvcc

FLAGS= `libpng-config --ldflags --cflags`
CXXFLAGS= --std=c++20
CUDAFLAGS=--std=c++17 -DGPU


CXXFILES=io.cpp main.cpp src/Classification.cpp src/ColorSimilarityMeasures.cpp src/TextureSimilarityMeasures.cpp src/TextureFeaturesExtraction.cpp
CUDAFILES=io.cpp -x cu main.cpp src/Classification.cu src/ColorSimilarityMeasures.cu src/TextureSimilarityMeasures.cu src/TextureFeaturesExtraction.cpp

gpu: $(CXXFILES)
	$(NVCC) -o gpgpu $(FLAGS) $(CUDAFLAGS) $(CUDAFILES)

cpu: $(CXXFILES)
	$(CC) -o gpgpu $(FLAGS) $(CXXFLAGS) $(CXXFILES) 

dlDataset:
	wget madrau.fr/dataset.zip
	unzip dataset.zip
	rm dataset.zip

clean:
	$(RM) gpgpu *.o
