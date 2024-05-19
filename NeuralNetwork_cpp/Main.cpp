#include "Network.h"
using namespace std;

int main() {
    // Define the model layers
    int modelLayers[] = {3, 4, 2}; // Example: 3 input neurons, 4 neurons in hidden layer, 2 output neurons
    cout<<"Hello";

    // Create a network with the specified model layers
    Network network(modelLayers, sizeof(modelLayers) / sizeof(modelLayers[0]));

    // Example: Perform forward propagation
    // Note: Replace the below code with your actual data and operations
    Neuron** weightsMatrix = nullptr; // Example: Get weights matrix from somewhere
    Neuron** inputMatrix = nullptr;   // Example: Get input matrix from somewhere
    Neuron* biasMatrix = nullptr;     // Example: Get bias matrix from somewhere
    int weightsRows = 0;               // Example: Set weights matrix rows
    int weightsCols = 0;               // Example: Set weights matrix columns
    int inputsRows = 0;                // Example: Set input matrix rows
    int inputCols = 0;                 // Example: Set input matrix columns

    Neuron** output = network.forwardPropagate(weightsMatrix, weightsRows, weightsCols, inputMatrix, inputsRows, inputCols, biasMatrix);

    // Example: Perform ReLU activation on the output
    // Note: Replace the below code with your actual activation function
    network.ReLU(output, weightsRows, inputCols);

    // Example: Cleanup and deallocate memory
    // Note: Replace the below code with your memory deallocation logic
    for (int i = 0; i < weightsRows; ++i) {
        delete[] weightsMatrix[i];
        delete[] inputMatrix[i];
        delete[] output[i];
    }
    delete[] weightsMatrix;
    delete[] inputMatrix;
    delete[] biasMatrix;
    delete[] output;

    return 0;
}
