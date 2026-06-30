cmake_minimum_required(VERSION 3.18)

project(MLPCUDA
  VERSION 1.0
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -flto")

add_executable(mlpCuda
  src/main.cpp
  src/matmulKernel.cu
  src/elementWiseKernel.cu
  src/mapKernel.cu
  src/sumColumnsKernel.cu
  src/biasBroadcastKernel.cu
)

target_include_directories(mlpCuda PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../data/include
)

target_link_libraries(mlpCuda PRIVATE cudart)
