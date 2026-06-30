#include "../include/cuda_wrappers.hpp"
#include <cuda_runtime.h>

__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ float relu(float x) { return fmaxf(0.0f, x); }
__device__ float sigmoidDerivada(float x) { return x * (1.0f - x); }
__device__ float reluDerivada(float x) { return x > 0.0f ? 1.0f : 0.0f; }

enum FuncType { SIGMOID, RELU, SIGMOID_DERIV, RELU_DERIV };

__global__ void mapKernel(float *A, int M, int N, FuncType func) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    switch (func) {
    case SIGMOID: {
      A[row * N + col] = sigmoid(A[row * N + col]);
      break;
    }
    case RELU: {
      A[row * N + col] = relu(A[row * N + col]);
      break;
    }
    case SIGMOID_DERIV: {
      A[row * N + col] = sigmoidDerivada(A[row * N + col]);
      break;
    }
    case RELU_DERIV: {
      A[row * N + col] = reluDerivada(A[row * N + col]);
      break;
    }
    }
  }
}

void mapSigmoidGpu(float *A, int M, int N) {

  dim3 blockDim(16, 16);
  dim3 gridDim((N + 15) / 16, (M + 15) / 16);

  mapKernel<<<gridDim, blockDim>>>(A, M, N, SIGMOID);

  cudaDeviceSynchronize();
}

void mapSigmoidDerivGpu(float *A, int M, int N) {

  dim3 blockDim(16, 16);
  dim3 gridDim((N + 15) / 16, (M + 15) / 16);

  mapKernel<<<gridDim, blockDim>>>(A, M, N, SIGMOID_DERIV);

  cudaDeviceSynchronize();
}

void mapReluGpu(float *A, int M, int N) {

  dim3 blockDim(16, 16);
  dim3 gridDim((N + 15) / 16, (M + 15) / 16);

  mapKernel<<<gridDim, blockDim>>>(A, M, N, RELU);

  cudaDeviceSynchronize();
}

void mapReluDerivGpu(float *A, int M, int N) {

  dim3 blockDim(16, 16);
  dim3 gridDim((N + 15) / 16, (M + 15) / 16);

  mapKernel<<<gridDim, blockDim>>>(A, M, N, RELU_DERIV);

  cudaDeviceSynchronize();
}
