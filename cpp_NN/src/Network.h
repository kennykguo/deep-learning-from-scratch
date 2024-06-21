#pragma once
#include <vector>
#include "LinearLayer.h"
#include "Neuron.h"

using namespace std;

class Network {
public:

    int numLayers; // Number of layers in the network

    vector<LinearLayer> networkLayers; // Vector of LinearLayer objects

    // Constructor (passing in the reference)
    Network(const vector<int>& modelLayers);

    // ReLU activation function (takes in a reference vector of vectors (a matrix))
    void ReLU(vector<vector<Neuron>>& matrix);

    // Derivative of ReLU activation function (takes in a reference vector of vectors (a matrix))
    void der_ReLU(vector<vector<Neuron>>& matrix);

    // Matrix multiplication function
    // Takes in two matrices of neurons and their corresponding rows and columns, and returns a matrice
    // Original matrices are not modified
    vector<vector<Neuron>> matrixMultiply(
        const vector<vector<Neuron>>& matrixOne, int rowsOne, int colsOne, 
        const vector<vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo);
};
