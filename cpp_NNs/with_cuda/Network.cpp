#include "Network.h"

using namespace std;

void Network::addLayer(std::unique_ptr<Layer> layer) {
    if (dynamic_cast<SoftmaxCrossEntropy*>(layer.get())) {
        lossLayer = dynamic_cast<SoftmaxCrossEntropy*>(layer.get());
    }
    layers.push_back(std::move(layer));
}

std::vector<std::vector<Neuron>> Network::forward(const std::vector<std::vector<Neuron>>& input, const std::vector<std::vector<Neuron>>& labels) {
    layerOutputs.clear();
    layerOutputs.push_back(input);

    std::vector<std::vector<Neuron>> current = input;
    for (const auto& layer : layers) {
        current = layer->forward(current);
        layerOutputs.push_back(current);
    }

    if (lossLayer) {
        lossLayer->setLabels(labels);
        loss = lossLayer->getLoss();
    }

    return current;
}

void Network::backward() {
    std::vector<std::vector<Neuron>> grad = layers.back()->backward(layerOutputs.back());

    for (int i = layers.size() - 2; i >= 0; --i) {
        grad = layers[i]->backward(grad);
    }
}

// void Network::setLabels(const vector<vector<Neuron>>& labels) {
//     if (lossLayer) {
//         lossLayer->setLabels(labels);
//     }
// }

// double Network::getLoss() const {
//     return lossLayer ? lossLayer->getLoss() : 0.0;
// }

vector<vector<Neuron>> Network::matrixMultiply(
    const vector<vector<Neuron>>& matrixOne, int rowsOne, int colsOne,
    const vector<vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo) {
    
    vector<vector<Neuron>> result;

    cudaMatrixMultiply(matrixOne, matrixTwo, result);

    return result;
}