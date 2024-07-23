#include "SoftmaxCrossEntropy.h"
#include <cmath>
#include <numeric>

using namespace std;

SoftmaxCrossEntropy::SoftmaxCrossEntropy(int numClasses) : numClasses(numClasses), loss(0) {}

vector<vector<Neuron>> SoftmaxCrossEntropy::forward(const vector<vector<Neuron>>& input) {
    lastInput = input;
    lastOutput.resize(input.size(), vector<Neuron>(numClasses));
    loss = 0;

    for (size_t i = 0; i < input.size(); ++i) {
        double maxVal = input[i][0].value;
        for (int j = 1; j < numClasses; ++j) {
            maxVal = max(maxVal, input[i][j].value);
        }

        double sum = 0;
        for (int j = 0; j < numClasses; ++j) {
            lastOutput[i][j].value = exp(input[i][j].value - maxVal);
            sum += lastOutput[i][j].value;
        }

        for (int j = 0; j < numClasses; ++j) {
            lastOutput[i][j].value /= sum;
            if (labels[i][j].value == 1) {
                loss -= log(lastOutput[i][j].value);
            }
        }
    }

    loss /= input.size();
    return lastOutput;
}

vector<vector<Neuron>> SoftmaxCrossEntropy::backward(const vector<vector<Neuron>>& gradOutput) {
    vector<vector<Neuron>> gradInput = lastOutput;
    for (size_t i = 0; i < gradInput.size(); ++i) {
        for (int j = 0; j < numClasses; ++j) {
            gradInput[i][j].value = (gradInput[i][j].value - labels[i][j].value) / gradInput.size();
        }
    }
    return gradInput;
}

void SoftmaxCrossEntropy::setLabels(const vector<vector<Neuron>>& labels) {
    this->labels = labels;
}