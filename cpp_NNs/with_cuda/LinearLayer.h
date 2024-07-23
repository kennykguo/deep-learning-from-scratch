#pragma once
#include "Layer.h"
using namespace std;

class LinearLayer : public Layer {
public:
    LinearLayer(int inputSize, int outputSize);
    
    vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) override;
    vector<vector<Neuron>> backward(const vector<vector<Neuron>>& gradOutput) override;

private:
    int inputSize;
    int outputSize;
    vector<vector<Neuron>> weights;
    vector<vector<Neuron>> lastInput;
};