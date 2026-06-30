#include "Activation.hpp"
#include "Matrix.hpp"
#include "cuda_wrappers.hpp"
#include <cmath>
#include <cstdlib>
#include <functional>
#include <random>

#pragma once

enum class ActivationType { RELU, SIGMOID };

template <typename T>

class Layer {
private:
  Matrix<T> weights;
  Matrix<T> bias;

  Matrix<T> inputData;
  Matrix<T> z;

  ActivationType activationType;

public:
  Layer(int neurons, int inputs, ActivationType type)
      : weights(neurons, inputs), bias(neurons, 1), activationType(type) {

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
    z = Matrix<T>(weights.getRows(), input.getColumns());

    matmulGpu(weights, input, z);
    biasBroadcastGpu(z, bias);

    Matrix<T> output = z;

    if (activationType == ActivationType::RELU)
      mapReluGpu(output);
    else
      mapSigmoidGpu(output);

    return output;
  }

  Matrix<T> calculateGradients(const Matrix<T> &outputError) {

    if (activationType == ActivationType::RELU)
      mapReluDerivGpu(z);
    else
      mapSigmoidDerivGpu(z);

    Matrix<T> gradients(z.getRows(), z.getColumns());

    hadamardGpu(z, outputError, gradients);

    return gradients;
  }

  Matrix<T> backward(const Matrix<T> &outputError, T learningRate) {

    Matrix<T> gradients = calculateGradients(outputError);

    // deltaWeights = (gradients * inputData^T) * lr
    Matrix<T> inputDataT(inputData.getColumns(), inputData.getRows());
    transposeGpu(inputData, inputDataT);

    Matrix<T> deltaWeights(weights.getRows(), weights.getColumns());
    matmulGpu(gradients, inputDataT, deltaWeights);
    scalarGpu(deltaWeights, deltaWeights, learningRate);

    // deltaBias = sumColumns(gradients) * lr
    Matrix<T> deltaBias(bias.getRows(), 1);
    sumColumnsGpu(gradients, deltaBias);
    scalarGpu(deltaBias, deltaBias, learningRate);

    // previousError = weights^T * gradients
    Matrix<T> weightsT(weights.getColumns(), weights.getRows());
    transposeGpu(weights, weightsT);

    Matrix<T> previousError(weights.getColumns(), gradients.getColumns());
    matmulGpu(weightsT, gradients, previousError);

    // weights += deltaWeights, bias += deltaBias
    addGpu(weights, deltaWeights, weights);
    addGpu(bias, deltaBias, bias);

    return previousError;
  }
};
