#include "Network.h"
#include <iostream>
#include <vector>

int main() {
    // Define matrix dimensions
    const int rowsOne = 3;
    const int colsOne = 2;
    const int rowsTwo = 2;
    const int colsTwo = 4;

    // Initialize matrices with random Neuron values
    std::vector<std::vector<Neuron>> matrixOne(rowsOne, std::vector<Neuron>(colsOne));
    std::vector<std::vector<Neuron>> matrixTwo(rowsTwo, std::vector<Neuron>(colsTwo));

    for (int i = 0; i < rowsOne; ++i) {
        for (int j = 0; j < colsOne; ++j) {
            matrixOne[i][j] = Neuron();
        }
    }

    for (int i = 0; i < rowsTwo; ++i) {
        for (int j = 0; j < colsTwo; ++j) {
            matrixTwo[i][j] = Neuron();
        }
    }

    // Create Network object
    Network network;

    // Perform matrix multiplication
    auto result = network.matrixMultiply(matrixOne, rowsOne, colsOne, matrixTwo, rowsTwo, colsTwo);

    // Output the result
    std::cout << "Result Matrix:\n";
    for (const auto& row : result) {
        for (const auto& neuron : row) {
            std::cout << neuron.value << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
