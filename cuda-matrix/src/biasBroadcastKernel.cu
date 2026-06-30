#include "../include/cuda_wrappers.hpp"
#include <cuda_runtime.h>

__global__ void biasBroadcastKernel(float *A, const float *bias, int M, int N) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    A[row * N + col] += bias[row];
  }
}

void biasBroadcastGpu(Matrix<float> &A, const Matrix<float> &bias) {

  int M = A.getRows(), N = A.getColumns();

  float *hostA = A.getData();
  const float *hostBias = bias.getData();

  float *deviceA = nullptr;
  float *deviceBias = nullptr;

  cudaMallocManaged(&deviceA, (M * N) * sizeof(float));
  cudaMallocManaged(&deviceBias, M * sizeof(float));

  cudaMemcpy(deviceA, hostA, (M * N) * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(deviceBias, hostBias, M * sizeof(float), cudaMemcpyDefault);

  dim3 blockDim(16, 16);
  dim3 gridDim((N + 15) / 16, (M + 15) / 16);

  biasBroadcastKernel<<<gridDim, blockDim>>>(deviceA, deviceBias, M, N);

  cudaDeviceSynchronize();

  cudaMemcpy(hostA, deviceA, (M * N) * sizeof(float), cudaMemcpyDefault);

  cudaFree(deviceA);
  cudaFree(deviceBias);
}
