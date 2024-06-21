#include "LinearLayer.h"
#include "Network.h"
#include <iostream>

using namespace std;

// Constructor
// Use of : initiates member variables before the constructor body executes
LinearLayer::LinearLayer(Network& network, int fan_in, int fan_out):network(network), fan_in(fan_in), fan_out(fan_out) {
    
    cout << "Created a LinearLayer" << endl;

    // Initialize weightsMatrix and biasMatrix
    // vector.resize() changes the size of the vector 
    weightsMatrix.resize(fan_out, vector<Neuron>(fan_in));
    biasMatrix.resize(fan_out);

    cout << "Weights Matrix Shape:\n";
    cout << "Rows: " << fan_in << '\n';
    cout << "Columns: " << fan_out << '\n';
}

// Forward propagation function
vector<vector<Neuron>> LinearLayer::forward(const vector<vector<Neuron>>& inputMatrix) {
    
    // Size gets the size of the corresponding vector
    // Vectors can be indexed like an array
    int inputsRows = inputMatrix.size();
    int inputCols = inputMatrix[0].size();
    
    cout << "Forward propagating:\n";
    cout << "Input Rows: " << inputsRows << "\n";
    cout << "Input Columns: " << inputCols << "\n";

    // Perform matrix multiplication
    vector<vector<Neuron>> output = network.matrixMultiply(inputMatrix, inputsRows, inputCols, weightsMatrix, fan_out, fan_in);
    
    // Add the bias
    for (int i = 0; i < fan_out; ++i) {
        for (int j = 0; j < inputCols; ++j) {
            output[i][j].value += biasMatrix[i].value;
        }
    }
    
    this->output_rows = inputsRows;
    this->output_cols = fan_out;
    return output;
}
