#include "Activation.hpp"
#include "Matrix.hpp"
#include <cmath>
#include <cstdlib>
#include <functional>
#include <random>

#pragma once

template <typename T>

class Layer {
private:
  Matrix<T> weights;
  Matrix<T> bias;

  Matrix<T> inputData;
  Matrix<T> z;

  std::function<T(T)> activationFunc;
  std::function<T(T)> derivativeFunc;

public:
  Layer(int neurons, int inputs, std::function<T(T)> activationFunc,
        std::function<T(T)> derivativeFunc)
      : weights(neurons, inputs), bias(neurons, 1),
        activationFunc(activationFunc), derivativeFunc(derivativeFunc) {

    T limit = std::sqrt(2.0 / inputs);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-limit, limit);

    weights.fill([&]() { return dis(gen); });
    bias.fill([]() { return (T)0; });
  }

  Matrix<T> &getWeights() { return weights; }
  Matrix<T> &getBias() { return bias; }

  Matrix<T> forward(const Matrix<T> &input) {

    inputData = input;

    Matrix<T> n = (weights * input) + bias;
    z = n;

    n.map(activationFunc);

    return n;
  }

  Matrix<T> calculateGradients(const Matrix<T> &outputError) {

    z.map(derivativeFunc);

    Matrix<T> gradients = z.hadamard(outputError);

    return gradients;
  }

  Matrix<T> backward(const Matrix<T> &outputError, T learningRate) {

    Matrix<T> gradients = calculateGradients(outputError);

    Matrix<T> deltaWeights = (gradients * inputData.Transpose()) * learningRate;

    Matrix<T> deltaBias = gradients * learningRate;

    Matrix<T> previousError = weights.Transpose() * gradients;

    weights = weights + deltaWeights;
    bias = bias + deltaBias;

    return previousError;
  }
};
