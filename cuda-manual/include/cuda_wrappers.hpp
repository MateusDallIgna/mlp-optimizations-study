#pragma once

// matmulKernel.cu
void matmulGpu(const float *A, const float *B, float *C, int M, int N, int K);

// elementWiseKernel.cu
void addGpu(const float *A, const float *B, float *C, int M, int N);
void subGpu(const float *A, const float *B, float *C, int M, int N);
void hadamardGpu(const float *A, const float *B, float *C, int M, int N);
void scalarGpu(const float *A, float *C, int M, int N, float scalar);

// mapKernel.cu
void mapReluGpu(float *A, int M, int N);
void mapReluDerivGpu(float *A, int M, int N);
void mapSigmoidGpu(float *A, int M, int N);
void mapSigmoidDerivGpu(float *A, int M, int N);

// sumColumnsKernel.cu
void sumColumnsGpu(const float *A, float *B, int M, int N);

// transposeKernel.cu
void transposeGpu(const float *A, float *B, int M, int N);

// biasBroadcastKernel.cu
void biasBroadcastGpu(float *A, const float *bias, int M, int N);
