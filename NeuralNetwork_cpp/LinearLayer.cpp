#include <cstdio>
#include <LinearLayer.h>
using namespace std;

// Default constructor
Layer::Layer(int fan_in, int fan_out) {
    this->fan_in = fan_in;
    this->fan_out = fan_out;

    // Create a 2D array for weightsMatrix
    weightsMatrix = new Neuron* [fan_out];
    for (int i = 0; i < fan_in; ++i) {
        weightsMatrix[i] = new Neuron[fan_in]; 
    }

    // Create a 1D array for biasMatrix
    biasMatrix = new Neuron[fan_out]; // (1, fan_out)
}

Layer::~Layer() {
    // Delete the arrays
    for (int i = 0; i < fan_in; ++i) {
        delete[] weightsMatrix[i];
    }
    delete[] weightsMatrix;
    delete[] biasMatrix;
}
