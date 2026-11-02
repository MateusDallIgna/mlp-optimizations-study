#include "../../data/include/mnist/mnist_reader.hpp"
#include "../include/Activation.hpp"
#include "../include/Layer.hpp"
#include "../include/Matrix.hpp"
#include "../include/Network.hpp"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

const int EPOCHS = 10;
const float LEARNING_RATE = 0.005;

auto dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../data");

int main() {

  struct Relu relu;
  struct Sigmoid sigmoid;
  Network<float> network;

  std::vector<Matrix<float>> inputs;
  std::vector<Matrix<float>> targets;
  std::vector<Matrix<float>> validationInputs;
  std::vector<Matrix<float>> validationTargets;

  // Creating layers, 2 hidden and one output
  // 128, 64, 10 neuros respectivtly
  network.addLayer(
      128, 784, [&relu](float x) { return relu(x); },
      [&relu](float x) { return relu.derivative(x); });

  network.addLayer(
      64, 128, [&relu](float x) { return relu(x); },
      [&relu](float x) { return relu.derivative(x); });

  network.addLayer(
      10, 64, [&sigmoid](float x) { return sigmoid(x); },
      [&sigmoid](float x) { return sigmoid.derivative(x); });

  // Adjusting the training dataset to the Matrix format
  for (int i = 0; i < 60000; i++) {
    Matrix<float> image(784, 1);
    const std::vector<uint8_t> &data = dataset.training_images[i];

    Matrix<float> target(10, 1);
    int label = dataset.training_labels[i];
    target(label, 0) = 1.0f;
    targets.push_back(target);

    for (int j = 0; j < 784; j++) {
      image(j, 0) = (float)data[j] / 255.0f;
    }

    inputs.push_back(image);
  }

  // Adjusting the validation dataset to the Matrix format
  for (int i = 0; i < 10000; i++) {
    Matrix<float> image(784, 1);
    const std::vector<uint8_t> &data = dataset.test_images[i];

    Matrix<float> target(10, 1);
    int label = dataset.test_labels[i];
    target(label, 0) = 1.0f;
    validationTargets.push_back(target);

    for (int j = 0; j < 784; j++) {
      image(j, 0) = (float)data[j] / 255.0f;
    }

    validationInputs.push_back(image);
  }

  std::chrono::time_point trainingBeginTime =
      std::chrono::high_resolution_clock::now();

  for (int i = 0; i < EPOCHS; i++) {
    for (int j = 0; j < inputs.size(); j++) {

      network.train(inputs[j], targets[j], LEARNING_RATE);
    }
  }

  std::chrono::time_point trainingEndTime =
      std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> trainingTime =
      trainingEndTime - trainingBeginTime;

  auto trainingThroughput = (inputs.size() * EPOCHS) / trainingTime.count();

  // Validation
  std::cout << "IMAGE_ID,REAL_LABEL,PREDICTED_LABEL,IS_CORRECT,CONFIDENCE\n";

  int rightGuesses = 0;

  std::vector<int> TP(10, 0);
  std::vector<int> FP(10, 0);
  std::vector<int> FN(10, 0);

  for (int i = 0; i < 10000; i++) {

    Matrix<float> anserw = network.forward(validationInputs[i]);
    int metricAnserw = dataset.test_labels[i];

    int networkGuess = 0;
    float highestProbability = anserw(0, 0);

    for (int j = 0; j < 10; j++) {
      if (anserw(j, 0) > highestProbability) {
        highestProbability = anserw(j, 0);
        networkGuess = j;
      }
    }

    int isCorrect = 0;
    if (networkGuess == dataset.test_labels[i]) {
      rightGuesses++;
      isCorrect = 1;

      TP[metricAnserw]++;

    } else {

      FP[networkGuess]++;
      FN[dataset.test_labels[i]]++;
    }

    std::cout << i << "," << (int)dataset.test_labels[i] << "," << networkGuess
              << "," << isCorrect << "," << std::fixed << std::setprecision(1)
              << (highestProbability * 100.0f) << "\n";
  }

  float accuracy = ((float)rightGuesses / 10000.0f) * 100.0f;

  std::vector<float> precision(10, 0);
  std::vector<float> recall(10, 0);

  for (int i = 0; i < 10; i++) {
    if (TP[i] != 0) {
      precision[i] = ((float)TP[i] / (TP[i] + FP[i])) * 100.0f;
      recall[i] = ((float)TP[i] / (TP[i] + FN[i])) * 100.0f;
    }
  }

  std::cout << "\n#--- FINAL METRICS ---" << std::endl;
  std::cout << "#Training Time: " << trainingTime.count() << " seconds"
            << std::endl;
  std::cout << "#Troughput: " << trainingThroughput << " samples/second"
            << std::endl;
  std::cout << "#Accuracy: " << accuracy << "%" << std::endl;

  for (int i = 0; i < 10; i++) {
    std::cout << "#Class " << i << " - Precision: " << std::fixed
              << std::setprecision(1) << precision[i]
              << "%, Recall:" << recall[i] << "%" << std::endl;
  }
}
