cmake_minimum_required(VERSION 3.22)
project(cudaMandelbrot CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(cudaMandelbrot main.cu)

find_package(OpenCV REQUIRED)

#set nvcc flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61 -ccbin=gcc-10")



set_target_properties(cudaMandelbrot PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cudaMandelbrot PRIVATE ${OpenCV_LIBS})

