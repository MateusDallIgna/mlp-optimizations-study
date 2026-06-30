#include "Activation.hpp"
#include "Matrix.hpp"
#include "cuda_wrappers.hpp"
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <functional>
#include <random>

#pragma once

enum class ActivationType { RELU, SIGMOID };

template <typename T>

class Layer {
private:
  float *weights;
  float *bias;
  float *z;
  float *inputData;
  float *gradients;
  float *deltaWeights;
  float *deltaBias;
  float *inputDataT;
  float *weightsT;
  float *previousError;
  int M;
  int K;

  ActivationType activationType;

public:
  Layer(const Layer &) = delete;
  Layer &operator=(const Layer &) = delete;

  Layer(Layer &&other) noexcept
      : weights(other.weights), bias(other.bias), z(other.z),
        inputData(other.inputData), gradients(other.gradients),
        deltaWeights(other.deltaWeights), deltaBias(other.deltaBias),
        inputDataT(other.inputDataT), weightsT(other.weightsT),
        previousError(other.previousError), M(other.M), K(other.K),
        activationType(other.activationType) {

    other.weights = nullptr;
    other.bias = nullptr;
    other.z = nullptr;
    other.inputData = nullptr;
    other.gradients = nullptr;
    other.deltaWeights = nullptr;
    other.deltaBias = nullptr;
    other.inputDataT = nullptr;
    other.weightsT = nullptr;
    other.previousError = nullptr;
  }
  Layer(int neurons, int inputs, ActivationType type, int batchSize)
      : activationType(type), M(neurons), K(inputs) {

    int N = batchSize; // batch size

    cudaMalloc(&weights, M * K * sizeof(float));
    cudaMalloc(&bias, M * sizeof(float));
    cudaMalloc(&z, M * N * sizeof(float));
    cudaMalloc(&inputData, K * N * sizeof(float));
    cudaMalloc(&gradients, M * N * sizeof(float));
    cudaMalloc(&deltaWeights, M * K * sizeof(float));
    cudaMalloc(&deltaBias, M * sizeof(float));
    cudaMalloc(&inputDataT, N * K * sizeof(float));
    cudaMalloc(&weightsT, K * M * sizeof(float));
    cudaMalloc(&previousError, K * N * sizeof(float));

    float *h_weights = new float[M * K];
    float *h_bias = new float[M]();

    T limit = std::sqrt(2.0 / inputs);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-limit, limit);

    for (int i = 0; i < M * K; i++)
      h_weights[i] = dis(gen);

    cudaMemcpy(weights, h_weights, M * K * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(bias, h_bias, M * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_weights;
    delete[] h_bias;
  }

  // Destructor
  ~Layer() {

    cudaFree(weights);
    cudaFree(bias);
    cudaFree(z);
    cudaFree(inputData);
    cudaFree(gradients);
    cudaFree(deltaWeights);
    cudaFree(deltaBias);
    cudaFree(inputDataT);
    cudaFree(weightsT);
    cudaFree(previousError);
  }

  float *getWeights() { return weights; }
  float *getBias() { return bias; }

  Matrix<T> forward(const Matrix<T> &input) {

    int N = input.getColumns();

    cudaMemcpy(inputData, input.getData(),
               (input.getColumns() * input.getRows()) * sizeof(float),
               cudaMemcpyHostToDevice);

    matmulGpu(weights, inputData, z, M, N, K);
    biasBroadcastGpu(z, bias, M, N);

    if (activationType == ActivationType::RELU)
      mapReluGpu(z, M, N);
    else
      mapSigmoidGpu(z, M, N);

    // coping back to Matrix
    Matrix<T> output(M, N);
    cudaMemcpy(output.getData(), z, M * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    return output;
  }

  void calculateGradients(const float *outputError, int N) {
    if (activationType == ActivationType::RELU)
      mapReluDerivGpu(z, M, N);
    else
      mapSigmoidDerivGpu(z, M, N);

    hadamardGpu(z, outputError, gradients, M, N);
  }

  float *backward(const float *outputError, int N, float learningRate) {

    calculateGradients(outputError, N);

    transposeGpu(inputData, inputDataT, K, N);
    matmulGpu(gradients, inputDataT, deltaWeights, M, K, N);
    scalarGpu(deltaWeights, deltaWeights, M, K, learningRate);

    sumColumnsGpu(gradients, deltaBias, M, N);
    scalarGpu(deltaBias, deltaBias, M, 1, learningRate);

    transposeGpu(weights, weightsT, M, K);
    matmulGpu(weightsT, gradients, previousError, K, N, M);

    addGpu(weights, deltaWeights, weights, M, K);
    addGpu(bias, deltaBias, bias, M, 1);

    return previousError;
  }
};
