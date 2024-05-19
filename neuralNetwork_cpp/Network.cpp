#include <LinearLayer.h>
#include <LinearLayer.cpp>
#include <cstdio>
#include <Network.h>
using namespace std;


Network::Network(int* modelLayers, int numLayers) {
    buffer = new char[sizeof(Layer) * numLayers];
    networkLayers = reinterpret_cast<Layer*>(buffer);
    for (int i = 0; i < numLayers - 1; i++) {
        new (&networkLayers[i]) Layer(modelLayers[i], modelLayers[i + 1]);
    }
    new (&networkLayers[numLayers - 1]) Layer(0, 0);
}


// Element-wise function
void Network::ReLU(Neuron** matrix, int rows, int cols)
{
    for (int i = 0; i<rows; i++)
    {
        for (int j = 0; j<cols; j++)
        {
            if (matrix[i][j].value <= 0)
            {
                matrix[i][j].value = 0;
            }
        }
    }
}


// Element-wise function
void Network::der_ReLU(Neuron** matrix, int rows, int cols)
{
    for (int i = 0; i<rows; i++)
    {
        for (int j = 0; j<cols; j++)
        {
            if (matrix[i][j].value <= 0)
            {
                matrix[i][j].value = 0;
            }
        }
    }
}


Neuron** Network::matrixMultiply(Neuron** matrixOne, int rowsOne, int colsOne, Neuron** matrixTwo, int rowsTwo, int colsTwo)
{
    int currentSum = 0;
    Neuron** result = new Neuron*[rowsOne];
    for (int i = 0; i < colsTwo; ++i) {
        result[i] = new Neuron[colsTwo]; 
    }
    for (int i = 0; i<rowsOne; i++)
    {
        for (int j = 0; i<colsTwo; j++)
        {
            currentSum = 0;
            for (int k = 0; k<colsOne; k++)
            {
                currentSum += matrixOne[i][k].value * matrixTwo[k][j].value;
            }
            (result[i][j]).value = currentSum;
        }
    }
}


Neuron** Network::forwardPropogate(Neuron** weightsMatrix, int weightsRows, int weightsCols, Neuron** inputMatrix, int inputsRows, int inputCols, Neuron* biasMatrix)
{
    Neuron **output = matrixMultiply(weightsMatrix, weightsRows, weightsCols, Neuron** inputMatrix, inputsRows, inputsCols)
    for (int i = 0; i<weightsRows; i++)
    {
        for (int j = 0; j<weightCols; j++)
        {
            output[i][j] += biasMatrix[j];
        }
    }
}