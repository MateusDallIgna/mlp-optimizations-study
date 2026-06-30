#include "../include/cuda_wrappers.hpp"
#include <cuda_runtime.h>

enum OpType { ADD, SUB, HADAMARD, SCALAR, TRANSPOSE };

__global__ void elementWiseKernel(const float *A, const float *B, float *C,
                                  int M, int N, OpType op, float scalar) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int xy = row * N + col;
  if (row < M && col < N) {
    switch (op) {
    case ADD: {
      C[xy] = A[xy] + B[xy];
      break;
    }
    case SUB: {
      C[xy] = A[xy] - B[xy];
      break;
    }
    case HADAMARD: {
      C[xy] = A[xy] * B[xy];
      break;
    }
    case SCALAR: {
      C[xy] = A[xy] * scalar;
      break;
    }
    case TRANSPOSE: {
      C[col * M + row] = A[row * N + col];
      break;
    }
    }
  }
}

void addGpu(const float *A, const float *B, float *C, int M, int N) {

  dim3 blockDim(16, 16);
  dim3 gridDim(((N + 15) / 16), ((M + 15) / 16));

  elementWiseKernel<<<gridDim, blockDim>>>(A, B, C, M, N, ADD, 0.0f);

  cudaDeviceSynchronize();
}

void subGpu(const float *A, const float *B, float *C, int M, int N) {

  dim3 blockDim(16, 16);
  dim3 gridDim(((N + 15) / 16), ((M + 15) / 16));

  elementWiseKernel<<<gridDim, blockDim>>>(A, B, C, M, N, SUB, 0.0f);

  cudaDeviceSynchronize();
}

void hadamardGpu(const float *A, const float *B, float *C, int M, int N) {

  dim3 blockDim(16, 16);
  dim3 gridDim(((N + 15) / 16), ((M + 15) / 16));

  elementWiseKernel<<<gridDim, blockDim>>>(A, B, C, M, N, HADAMARD, 0.0f);

  cudaDeviceSynchronize();
}

void scalarGpu(const float *A, float *C, int M, int N, float scalar) {

  dim3 blockDim(16, 16);
  dim3 gridDim(((N + 15) / 16), ((M + 15) / 16));

  elementWiseKernel<<<gridDim, blockDim>>>(A, nullptr, C, M, N, SCALAR,
                                           scalar);

  cudaDeviceSynchronize();
}

void transposeGpu(const float *A, float *B, int M, int N) {

  dim3 blockDim(16, 16);
  dim3 gridDim(((N + 15) / 16), ((M + 15) / 16));

  elementWiseKernel<<<gridDim, blockDim>>>(A, nullptr, B, M, N, TRANSPOSE,
                                           0.0f);

  cudaDeviceSynchronize();
}
