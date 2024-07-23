#include "LinearLayer.h"
#include <random>
#include <cmath>

using namespace std;

LinearLayer::LinearLayer(int inputSize, int outputSize) 
    : inputSize(inputSize), outputSize(outputSize) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0, 1);

    weights.resize(inputSize, vector<Neuron>(outputSize));
    for (auto& row : weights) {
        for (auto& weight : row) {
            weight.value = d(gen) * sqrt(2.0 / inputSize);
        }
    }
}

vector<vector<Neuron>> LinearLayer::forward(const vector<vector<Neuron>>& input) {
    lastInput = input;
    vector<vector<Neuron>> output(input.size(), vector<Neuron>(outputSize));

    for (size_t i = 0; i < input.size(); ++i) {
        for (int j = 0; j < outputSize; ++j) {
            double sum = 0;
            for (int k = 0; k < inputSize; ++k) {
                sum += input[i][k].value * weights[k][j].value;
            }
            output[i][j].value = sum;
        }
    }

    return output;
}

vector<vector<Neuron>> LinearLayer::backward(const vector<vector<Neuron>>& gradOutput) {
    vector<vector<Neuron>> gradInput(lastInput.size(), vector<Neuron>(inputSize));

    for (size_t i = 0; i < lastInput.size(); ++i) {
        for (int j = 0; j < inputSize; ++j) {
            double sum = 0;
            for (int k = 0; k < outputSize; ++k) {
                sum += gradOutput[i][k].value * weights[j][k].value;
                weights[j][k].value -= 0.01 * gradOutput[i][k].value * lastInput[i][j].value; // Simple SGD update
            }
            gradInput[i][j].value = sum;
        }
    }
    return gradInput;
}