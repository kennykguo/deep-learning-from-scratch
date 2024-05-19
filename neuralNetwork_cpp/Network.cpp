#include <linearLayer.cpp>
#include <cstdio>
#include <Network.h>
using namespace std;

Network::Network(int* numLayers)
{

    networkLayers = 
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
