#include "../include/cuda_wrappers.hpp"
#include <cuda_runtime.h>

__global__ void sumColumnsKernel(const float *A, float *B, int M, int N) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M) {

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
      sum += A[row * N + j];
    }
    B[row] = sum;
  }
}

void sumColumnsGpu(const float *A, float *B, int M, int N) {

  dim3 blockDim(256);
  dim3 gridDim((M + 255) / 256);

  sumColumnsKernel<<<gridDim, blockDim>>>(A, B, M, N);

  cudaDeviceSynchronize();
}
