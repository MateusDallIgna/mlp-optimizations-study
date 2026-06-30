#include "../include/cuda_wrappers.hpp"
#include <cuda_runtime.h>

__global__ void biasBroadcastKernel(float *A, const float *bias, int M, int N) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    A[row * N + col] += bias[row];
  }
}

void biasBroadcastGpu(float *A, const float *bias, int M, int N) {

  dim3 blockDim(16, 16);
  dim3 gridDim((N + 15) / 16, (M + 15) / 16);

  biasBroadcastKernel<<<gridDim, blockDim>>>(A, bias, M, N);

  cudaDeviceSynchronize();
}
