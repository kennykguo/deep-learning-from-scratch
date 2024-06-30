#pragma once

#include "Neuron.h"
#include <vector>

class Network {
public:
    void ReLU(std::vector<std::vector<Neuron>>& matrix);
    void der_ReLU(std::vector<std::vector<Neuron>>& matrix);
    std::vector<std::vector<Neuron>> matrixMultiply(
        const std::vector<std::vector<Neuron>>& matrixOne, int rowsOne, int colsOne,
        const std::vector<std::vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo);
};
