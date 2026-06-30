#include "../include/cuda_wrappers.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t g_cublasHandle = nullptr;

void matmulGpu(const float *A, const float *B, float *C,
               int M, int N, int K) {

  if (!g_cublasHandle) {
    cublasCreate(&g_cublasHandle);
    cublasSetMathMode(g_cublasHandle, CUBLAS_TF32_TENSOR_OP_MATH);
  }

  const float alpha = 1.0f, beta = 0.0f;

  // Row-major: C[M×N] = A[M×K] * B[K×N]
  // Column-major (cuBLAS): C^T[N×M] = B^T[N×K] * A^T[K×M]
  cublasSgemm(g_cublasHandle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K,
              &alpha,
              B, N,
              A, K,
              &beta,
              C, N);
}
