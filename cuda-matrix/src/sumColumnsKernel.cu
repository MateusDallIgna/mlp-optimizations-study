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

void sumColumnsGpu(const Matrix<float> &A, Matrix<float> &B) {

  int M = A.getRows(), N = A.getColumns();

  // Getting the host(Cpu) data
  const float *hostA = A.getData();
  float *hostB = B.getData();

  float *deviceA = nullptr;
  float *deviceB = nullptr;

  cudaMallocManaged(&deviceA, (M * N) * sizeof(float));
  cudaMallocManaged(&deviceB, (M * 1) * sizeof(float));

  cudaMemcpy(deviceA, hostA, (M * N) * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(deviceB, hostB, M * sizeof(float), cudaMemcpyDefault);

  dim3 blockDim(256);
  dim3 gridDim((M + 255) / 256);

  sumColumnsKernel<<<gridDim, blockDim>>>(deviceA, deviceB, M, N);

  cudaDeviceSynchronize();

  cudaMemcpy(hostB, deviceB, M * sizeof(float), cudaMemcpyDefault);

  cudaFree(deviceA);
  cudaFree(deviceB);
}
