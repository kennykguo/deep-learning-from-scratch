#include "Network.h"
#include <iostream>
#include <cassert>

void Network::ReLU(std::vector<std::vector<Neuron>>& matrix) {
    for (auto& row : matrix) {
        for (auto& neuron : row) {
            if (neuron.value <= 0) {
                neuron.value = 0;
            }
        }
    }
}

void Network::der_ReLU(std::vector<std::vector<Neuron>>& matrix) {
    for (auto& row : matrix) {
        for (auto& neuron : row) {
            if (neuron.value <= 0) {
                neuron.value = 0;
            }
        }
    }
}

std::vector<std::vector<Neuron>> Network::matrixMultiply(
    const std::vector<std::vector<Neuron>>& matrixOne, int rowsOne, int colsOne,
    const std::vector<std::vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo) {

    assert(colsOne == rowsTwo && "Matrix dimensions must match for multiplication.");

    std::vector<std::vector<Neuron>> result(rowsOne, std::vector<Neuron>(colsTwo));

    for (int i = 0; i < rowsOne; ++i) {
        for (int j = 0; j < colsTwo; ++j) {
            double currentSum = 0.0;
            for (int k = 0; k < colsOne; ++k) {
                currentSum += matrixOne[i][k].value * matrixTwo[k][j].value;
            }
            result[i][j].value = currentSum;
        }
    }

    return result;
}
