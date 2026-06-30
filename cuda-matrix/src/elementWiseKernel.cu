#include "../include/cuda_wrappers.hpp"
#include <cuda_runtime.h>

enum OpType { ADD, SUB, HADAMARD, SCALAR, TRANSPOSE };

__global__ void elementWiseKernel(float *A, float *B, float *C, int M, int N,
                                  OpType op, float scalar) {

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

void addGpu(const Matrix<float> &A, const Matrix<float> &B, Matrix<float> &C) {

  int M = A.getRows(), N = A.getColumns();

  // Getting the host(Cpu) data
  const float *hostA = A.getData();
  const float *hostB = B.getData();
  float *hostC = C.getData();

  float *deviceA = nullptr;
  float *deviceB = nullptr;
  float *deviceC = nullptr;

  cudaMallocManaged(&deviceA, (M * N) * sizeof(float));
  cudaMallocManaged(&deviceB, (M * N) * sizeof(float));
  cudaMallocManaged(&deviceC, (M * N) * sizeof(float));

  cudaMemcpy(deviceA, hostA, (M * N) * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(deviceB, hostB, (M * N) * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(deviceC, hostC, (M * N) * sizeof(float), cudaMemcpyDefault);

  dim3 blockDim(16, 16);
  dim3 gridDim(((N + 15) / 16), ((M + 15) / 16));

  elementWiseKernel<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, M, N, ADD,
                                           0.0f);

  cudaDeviceSynchronize();

  cudaMemcpy(hostC, deviceC, (M * N) * sizeof(float), cudaMemcpyDefault);

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
}

void subGpu(const Matrix<float> &A, const Matrix<float> &B, Matrix<float> &C) {

  int M = A.getRows(), N = A.getColumns();

  // Getting the host(Cpu) data
  const float *hostA = A.getData();
  const float *hostB = B.getData();
  float *hostC = C.getData();

  float *deviceA = nullptr;
  float *deviceB = nullptr;
  float *deviceC = nullptr;

  cudaMallocManaged(&deviceA, (M * N) * sizeof(float));
  cudaMallocManaged(&deviceB, (M * N) * sizeof(float));
  cudaMallocManaged(&deviceC, (M * N) * sizeof(float));

  cudaMemcpy(deviceA, hostA, (M * N) * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(deviceB, hostB, (M * N) * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(deviceC, hostC, (M * N) * sizeof(float), cudaMemcpyDefault);

  dim3 blockDim(16, 16);
  dim3 gridDim(((N + 15) / 16), ((M + 15) / 16));

  elementWiseKernel<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, M, N, SUB,
                                           0.0f);

  cudaDeviceSynchronize();

  cudaMemcpy(hostC, deviceC, (M * N) * sizeof(float), cudaMemcpyDefault);

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
}

void hadamardGpu(const Matrix<float> &A, const Matrix<float> &B,
                 Matrix<float> &C) {

  int M = A.getRows(), N = A.getColumns();

  // Getting the host(Cpu) data
  const float *hostA = A.getData();
  const float *hostB = B.getData();
  float *hostC = C.getData();

  float *deviceA = nullptr;
  float *deviceB = nullptr;
  float *deviceC = nullptr;

  cudaMallocManaged(&deviceA, (M * N) * sizeof(float));
  cudaMallocManaged(&deviceB, (M * N) * sizeof(float));
  cudaMallocManaged(&deviceC, (M * N) * sizeof(float));

  cudaMemcpy(deviceA, hostA, (M * N) * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(deviceB, hostB, (M * N) * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(deviceC, hostC, (M * N) * sizeof(float), cudaMemcpyDefault);

  dim3 blockDim(16, 16);
  dim3 gridDim(((N + 15) / 16), ((M + 15) / 16));

  elementWiseKernel<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, M, N,
                                           HADAMARD, 0.0f);

  cudaDeviceSynchronize();

  cudaMemcpy(hostC, deviceC, (M * N) * sizeof(float), cudaMemcpyDefault);

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
}
void scalarGpu(const Matrix<float> &A, Matrix<float> &C, float scalar) {

  int M = A.getRows(), N = A.getColumns();

  // Getting the host(Cpu) data
  const float *hostA = A.getData();
  float *hostC = C.getData();

  float *deviceA = nullptr;
  float *deviceC = nullptr;

  cudaMallocManaged(&deviceA, (M * N) * sizeof(float));
  cudaMallocManaged(&deviceC, (M * N) * sizeof(float));

  cudaMemcpy(deviceA, hostA, (M * N) * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(deviceC, hostC, (M * N) * sizeof(float), cudaMemcpyDefault);

  dim3 blockDim(16, 16);
  dim3 gridDim(((N + 15) / 16), ((M + 15) / 16));

  elementWiseKernel<<<gridDim, blockDim>>>(deviceA, nullptr, deviceC, M, N,
                                           SCALAR, scalar);

  cudaDeviceSynchronize();

  cudaMemcpy(hostC, deviceC, (M * N) * sizeof(float), cudaMemcpyDefault);

  cudaFree(deviceA);
  cudaFree(deviceC);
}

void transposeGpu(const Matrix<float> &A, Matrix<float> &B) {

  int M = A.getRows(), N = A.getColumns();

  const float *hostA = A.getData();
  float *hostB = B.getData();

  float *deviceA = nullptr;
  float *deviceB = nullptr;

  cudaMallocManaged(&deviceA, (M * N) * sizeof(float));
  cudaMallocManaged(&deviceB, (M * N) * sizeof(float));

  cudaMemcpy(deviceA, hostA, (M * N) * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(deviceB, hostB, (M * N) * sizeof(float), cudaMemcpyDefault);

  dim3 blockDim(16, 16);
  dim3 gridDim(((N + 15) / 16), ((M + 15) / 16));

  elementWiseKernel<<<gridDim, blockDim>>>(deviceA, nullptr, deviceB, M, N,
                                           TRANSPOSE, 0.0f);

  cudaDeviceSynchronize();

  cudaMemcpy(hostB, deviceB, (M * N) * sizeof(float), cudaMemcpyDefault);

  cudaFree(deviceA);
  cudaFree(deviceB);
}
