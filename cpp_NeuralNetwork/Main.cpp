#include <iostream> // Include for cout
#include "Network.h"

int main() {
    // Define the model layers
    int modelLayers[] = {3, 4, 2}; // Example: 3 input neurons, 4 neurons in hidden layer, 2 output neurons

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

    // Example: Print out weightsMatrix
    std::cout << "Weights Matrix:\n";
    for (int i = 0; i < weightsRows; ++i) {
        for (int j = 0; j < weightsCols; ++j) {
            std::cout << weightsMatrix[i][j].value << " ";
        }
        std::cout << "\n";
    }

    // Example: Print out inputMatrix
    std::cout << "Input Matrix:\n";
    for (int i = 0; i < inputsRows; ++i) {
        for (int j = 0; j < inputCols; ++j) {
            std::cout << inputMatrix[i][j].value << " ";
        }
        std::cout << "\n";
    }

    // Example: Print out output
    std::cout << "Output Matrix:\n";
    for (int i = 0; i < weightsRows; ++i) {
        for (int j = 0; j < inputCols; ++j) {
            std::cout << output[i][j].value << " ";
        }
        std::cout << "\n";
    }

    // Cleanup (delete dynamic memory, etc.)
    // Note: Replace with actual cleanup code

    return 0;
}
