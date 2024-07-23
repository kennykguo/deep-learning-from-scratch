#pragma once
#include <vector>
#include <memory>
#include "Layer.h"
#include "SoftmaxCrossEntropy.h"

using namespace std;

class Network {
public:
    void addLayer(std::unique_ptr<Layer> layer);
    std::vector<std::vector<Neuron>> forward(const std::vector<std::vector<Neuron>>& input, const std::vector<std::vector<Neuron>>& labels);
    void backward();
    double getLoss() const { return loss; }
    // Add the matrixMultiply function declaration
    static vector<vector<Neuron>> matrixMultiply(
        const vector<vector<Neuron>>& matrixOne, int rowsOne, int colsOne,
        const vector<vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo);

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<std::vector<std::vector<Neuron>>> layerOutputs;
    SoftmaxCrossEntropy* lossLayer;
    double loss;
};


extern "C" void cudaMatrixMultiply(const std::vector<std::vector<Neuron>>& A, 
                                   const std::vector<std::vector<Neuron>>& B, 
                                   std::vector<std::vector<Neuron>>& C);