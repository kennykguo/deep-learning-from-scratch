#include "Network.h"
#include <iostream>

using namespace std;

// Constructor
Network::Network(const vector<int>& modelLayers) {

    cout << "Created a Network!" << '\n';

    // Gets the size of the modelLayers vector
    this-> numLayers = modelLayers.size();

    // Loop through each layer (excluding the last element of modelLayers)
    for (int i = 0; i < numLayers - 1; i++) {
        // Creates a linear layer for each pair of consecutive modelLayer values
        // Constructs an element directly in the vector
        networkLayers.emplace_back(*this, modelLayers[i], modelLayers[i + 1]);
    }
    // Adds a dummy layer at the end to handle the case where they might be no more layers to process
    networkLayers.emplace_back(*this, 0, 0);
}

// ReLU activation function
void Network::ReLU(vector<vector<Neuron>>& matrix) {
    // Loops over every row and every neuron in the row
    // Auto allows the compiler to automatically determine the type of a variable
    // Loops through every vector of vectors in matrix (every row)
    // The : convention is the range based loop -> 
    for (vector<Neuron>& row : matrix) {
        // Iterate through each neuron in the row
        for (Neuron& neuron : row) {
            // Apply ReLU activation function: set value to 0 if it's less than or equal to 0
            if (neuron.value <= 0) {
                neuron.value = 0;
            }
        }
    }
}

// Derivative of ReLU activation function
void Network::der_ReLU(vector<vector<Neuron>>& matrix) {
    for (vector<Neuron>& row : matrix) {
        // Iterate through each neuron in the row
        for (Neuron& neuron : row) {
            // Apply the derivative of ReLU: set value to 0 if it's less than or equal to 0
            if (neuron.value <= 0) {
                neuron.value = 0;
            }
        }
    }
}

// Matrix multiplication function
vector<vector<Neuron>> Network::matrixMultiply(
    const vector<vector<Neuron>>& matrixOne, int rowsOne, int colsOne, 
    const vector<vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo) {

    vector<vector<Neuron>> result(rowsOne, vector<Neuron>(colsTwo));
    
    // colsOne = rowsTwo
    for (int i = 0; i < rowsOne; ++i) {
        for (int j = 0; j < colsTwo; ++j) {
            int currentSum = 0;
            // Number of entries to sum up for
            for (int k = 0; k < colsOne; ++k) {
                currentSum += matrixOne[i][k].value * matrixTwo[k][j].value;
            }
            result[i][j].value = currentSum;
        }
    }
    return result;
}
