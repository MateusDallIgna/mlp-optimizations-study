#include "../include/cuda_wrappers.hpp"
#include <cuda_runtime.h>

__global__ void matmulKernel(const float *A, const float *B, float *C, int M,
                             int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

void matmulGpu(const float *A, const float *B, float *C, int M, int N, int K) {

  dim3 blockDim(16, 16);
  dim3 gridDim(((N + 15) / 16), ((M + 15) / 16));

  matmulKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);

  cudaDeviceSynchronize();
}
