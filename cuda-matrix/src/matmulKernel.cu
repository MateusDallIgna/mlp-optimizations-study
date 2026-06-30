#include "../include/cuda_wrappers.hpp"
#include <cuda_runtime.h>

// A[M][K] * B[K][N] = C[M][N]
__global__ void matmulKernel(float *A, float *B, float *C, int M, int N,
                             int K) {
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

void matmulGpu(const Matrix<float> &A, const Matrix<float> &B,
               Matrix<float> &C) {

  int M = A.getRows(), N = B.getColumns(), K = A.getColumns();

  // Getting the host(Cpu) data
  const float *hostA = A.getData();
  const float *hostB = B.getData();
  float *hostC = C.getData();

  float *deviceA = nullptr;
  float *deviceB = nullptr;
  float *deviceC = nullptr;

  cudaMallocManaged(&deviceA, (M * K) * sizeof(float));
  cudaMallocManaged(&deviceB, (K * N) * sizeof(float));
  cudaMallocManaged(&deviceC, (M * N) * sizeof(float));

  cudaMemcpy(deviceA, hostA, (M * K) * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(deviceB, hostB, (K * N) * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(deviceC, hostC, (M * N) * sizeof(float), cudaMemcpyDefault);

  dim3 blockDim(16, 16);
  // How many blocks of 16 threads each are needed to fill the gridDim
  dim3 gridDim(((N + 15) / 16), ((M + 15) / 16));

  matmulKernel<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, M, N, K);

  cudaDeviceSynchronize();

  cudaMemcpy(hostC, deviceC, (M * N) * sizeof(float), cudaMemcpyDefault);

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
}
