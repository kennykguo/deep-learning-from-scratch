
#include "LinearLayer.h"
#include "Network.h"
#include "Neuron.h"


class Network;

// Default constructor
Layer::Layer(Network& network, int fan_in, int fan_out)
    : network(network), fan_in(fan_in), fan_out(fan_out) {
    // this->fan_in = fan_in;
    // this->fan_out = fan_out;

    // Create a 2D array for weightsMatrix
    weightsMatrix = new Neuron* [fan_out];
    for (int i = 0; i < fan_in; ++i) {
        weightsMatrix[i] = new Neuron[fan_in]; 
    }

    // Create a 1D array for biasMatrix
    biasMatrix = new Neuron[fan_out]; // (1, fan_out)
}




