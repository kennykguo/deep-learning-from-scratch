#include "Network.h"
#include <iostream>
#include <cassert>
#include <iostream>

using namespace std;


// The Network classes define the architecture, and you must choose the layers before you compile the program


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



vector<vector<Neuron>> Network::matrixMultiply(
    const vector<vector<Neuron>>& matrixOne, int rowsOne, int colsOne,
    const vector<vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo) {

    // Ensure the dimensions are correct
    assert(colsOne == rowsTwo && "Matrix dimensions must match for multiplication.");

    // Initialize result matrix with the correct dimensions
    vector<vector<Neuron>> result(rowsOne, vector<Neuron>(colsTwo));

    // Perform the matrix multiplication
    for (int i = 0; i < rowsOne; ++i) {
        for (int j = 0; j < colsTwo; ++j) {
            double currentSum = 0.0;
            for (int k = 0; k < colsOne - 1; ++k) {
                // Additional checks and logging
                assert(i < matrixOne.size());
                assert(k < matrixOne[i].size());
                assert(k < matrixTwo.size());
                assert(j < matrixTwo[k].size());

                // Debugging information
                // K is out of bounds!
                // Check GPT for kgkgg
                if (!(i < matrixOne.size())) cerr << "i out of bounds: " << i << " >= " << matrixOne.size() << endl;
                if (!(k < matrixOne[i].size())) cerr << "k out of bounds for matrixOne: " << k << " >= " << matrixOne[i].size() << endl;
                if (!(k < matrixTwo.size())) cerr << "k out of bounds for matrixTwo: " << k << " >= " << matrixTwo.size() << endl;
                if (!(j < matrixTwo[k].size())) cerr << "j out of bounds for matrixTwo: " << j << " >= " << matrixTwo[k].size() << endl;

                currentSum += matrixOne[i][k].value * matrixTwo[k][j].value;
            }
            result[i][j].value = currentSum;
        }
    }

    return result;
}



// // Matrix multiplication function
// vector<vector<Neuron>> Network::matrixMultiply(
//     const vector<vector<Neuron>>& matrixOne, int rowsOne, int colsOne, 
//     const vector<vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo) {

//     cout << "Matrix multiplication:\n";
//     cout << "MatrixOne Rows: " << rowsOne << "\n";
//     cout << "MatrixOne Columns: " << colsOne << "\n";
//     cout << "MatrixTwo Rows: " << rowsTwo << "\n";
//     cout << "MatrixTwo Columns: " << colsTwo << "\n";

//     assert(colsOne == rowsTwo && "Matrix dimensions must match for multiplication.");

//     // Initialize result matrix with the correct dimensions
//     vector<vector<Neuron>> result(rowsOne, vector<Neuron>(colsTwo));
//     cout << "Initialized result matrix with dimensions: " << result.size() << " x " << (result.empty() ? 0 : result[0].size()) << "\n";

//     // Perform the matrix multiplication
//     for (int i = 0; i < rowsOne; ++i) {
//         for (int j = 0; j < colsTwo; ++j) {
//             double currentSum = 0.0;
//             for (int k = 0; k < colsOne; ++k) {
//                 currentSum += matrixOne[i][k].value * matrixTwo[k][j].value;
//             }
//             result[i][j].value = currentSum;
//         }
//     }

//     // Debugging: print some values from the result to ensure the computation was performed
//     cout << "Sample values from the result matrix:\n";
//     if (!result.empty() && !result[0].empty()) {
//         cout << "result[0][0].value: " << result[0][0].value << "\n";
//         cout << "result[0][1].value: " << result[0][1].value << "\n";
//     }

//     cout << "Result Rows: " << result.size() << "\n";
//     if (!result.empty()) {
//         cout << "Result Columns: " << result[0].size() << "\n";
//     }
//     return result;
// }



