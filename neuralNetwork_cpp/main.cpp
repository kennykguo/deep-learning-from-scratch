#include <iostream>
#include "Network.h" // Include Network header

int main() {
    // Define your network architecture
    int modelLayers[] = {3, 4, 5}; // Example architecture: 3 input neurons, 4 neurons in hidden layer, 5 output neurons
    int numLayers = sizeof(modelLayers) / sizeof(modelLayers[0]);

    // Create a Network object
    Network myNetwork(modelLayers, numLayers);

    // Test your network by calling its functions
    // Example: Call ReLU function
    // Neuron** exampleMatrix = new Neuron*[3]; // Example input matrix with 3 rows
    // for (int i = 0; i < 3; ++i) {
    //     exampleMatrix[i] = new Neuron[4]; // Example input matrix with 4 columns
    // }
    // myNetwork.ReLU(exampleMatrix, 3, 4); // Example call to ReLU function

    // Perform other tests or operations as needed...

    return 0;
}
