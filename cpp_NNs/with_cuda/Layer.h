#pragma once
#include <vector>
#include "Neuron.h"

using namespace std;

class Layer {
public:
    virtual vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) = 0;
    virtual vector<vector<Neuron>> backward(const vector<vector<Neuron>>& gradOutput) = 0;
    virtual ~Layer() = default;
};