#pragma once
#include "Layer.h"

using namespace std;

class SoftmaxCrossEntropy : public Layer {
public:
    SoftmaxCrossEntropy(int numClasses);
    
    vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) override;
    vector<vector<Neuron>> backward(const vector<vector<Neuron>>& gradOutput) override;
    
    void setLabels(const vector<vector<Neuron>>& labels);
    double getLoss() const { return loss; }

private:
    int numClasses;
    vector<vector<Neuron>> lastInput;
    vector<vector<Neuron>> lastOutput;
    vector<vector<Neuron>> labels;
    double loss;
};