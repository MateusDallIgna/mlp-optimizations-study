#pragma once
#include "Matrix.hpp"

// matmulKernel.cu
void matmulGpu(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);

// elementWiseKernel.cu
void addGpu(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
void subGpu(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
void hadamardGpu(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
void scalarGpu(const Matrix<float>& A, Matrix<float>& C, float scalar);

// mapKernel.cu
void mapReluGpu(Matrix<float>& A);
void mapReluDerivGpu(Matrix<float>& A);
void mapSigmoidGpu(Matrix<float>& A);
void mapSigmoidDerivGpu(Matrix<float>& A);

// sumColumnsKernel.cu
void sumColumnsGpu(const Matrix<float>& A, Matrix<float>& B);

// transposeKernel.cu
void transposeGpu(const Matrix<float>& A, Matrix<float>& B);

// biasBroadcastKernel.cu
void biasBroadcastGpu(Matrix<float>& A, const Matrix<float>& bias);
