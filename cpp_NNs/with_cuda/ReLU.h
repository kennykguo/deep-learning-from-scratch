#pragma once
#include "Layer.h"

using namespace std;

class ReLU : public Layer {
public:
    vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) override;
    vector<vector<Neuron>> backward(const vector<vector<Neuron>>& gradOutput) override;

private:
    vector<vector<Neuron>> lastInput;
};