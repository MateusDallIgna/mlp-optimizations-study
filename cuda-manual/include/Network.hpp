#include "Layer.hpp"
#include "Matrix.hpp"
#include "cuda_wrappers.hpp"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#pragma once

template <typename T>

class Network {

private:
  std::vector<Layer<T>> networkLayers;
  float *error;
  float *errorGpu;
  int batchSize;

public:
  Network(int batchSize) : batchSize(batchSize) {
    cudaMalloc(&error, 10 * batchSize * sizeof(float));
    cudaMalloc(&errorGpu, 10 * batchSize * sizeof(float));
  };

  ~Network() {
    cudaFree(error);
    cudaFree(errorGpu);
  }
  void addLayer(int neurons, int inputs, ActivationType type) {

    networkLayers.push_back(Layer<T>(neurons, inputs, type, batchSize));
  };

  Matrix<T> forward(Matrix<T> inputs) {
    for (int i = 0; i < networkLayers.size(); i++) {
      inputs = networkLayers[i].forward(inputs);
    }
    return inputs;
  }

  void backward(float *outputError, int N, T learningRate) {
    for (int i = networkLayers.size() - 1; i >= 0; i--)
      outputError = networkLayers[i].backward(outputError, N, learningRate);
  }

  void train(Matrix<T> &input, Matrix<T> targets, T learningRate) {

    Matrix<T> predictions = forward(input);

    int M = targets.getRows(), N = targets.getColumns();

    cudaMemcpy(error, targets.getData(), M * N * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(errorGpu, predictions.getData(), M * N * sizeof(float),
               cudaMemcpyHostToDevice);

    subGpu(error, errorGpu, error, M, N);

    float invN = 1.0f / N;
    scalarGpu(error, error, M, N, invN);

    backward(error, N, learningRate);
  }
};
