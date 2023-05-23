cmake_minimum_required(VERSION 3.24)
project(Project CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(Project test.cpp main.cu)

set_target_properties(Project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
