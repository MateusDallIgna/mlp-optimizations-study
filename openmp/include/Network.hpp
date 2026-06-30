#include "Layer.hpp"
#include "Matrix.hpp"
#include <vector>
#pragma once

template <typename T>

class Network {

private:
  std::vector<Layer<T>> networkLayers;

public:
  void addLayer(int neurons, int inputs, std::function<T(T)> activationFunc,
                std::function<T(T)> derivativeFunc) {

    networkLayers.push_back(
        Layer<T>(neurons, inputs, activationFunc, derivativeFunc));
  };

  Matrix<T> forward(Matrix<T> inputs) {
    for (int i = 0; i < networkLayers.size(); i++) {
      inputs = networkLayers[i].forward(inputs);
    }
    return inputs;
  }

  Matrix<T> backward(Matrix<T> outputError, T learningRate) {
    for (int i = networkLayers.size() - 1; i >= 0; i--) {

      outputError = networkLayers[i].backward(outputError, learningRate);
    }
    return outputError;
  };

  void train(Matrix<T> &input, Matrix<T> targets, T learningRate) {

    Matrix<T> predictions = forward(input);
    Matrix<T> error = targets - predictions;
    backward(error, learningRate);
  }
};
