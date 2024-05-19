#include <LinearLayer.cpp>
#include <cstdio>
using namespace std;

class Network
{
public:
    int numLayers;
    Layer* networkLayers;
    char* buffer;

    Network(int* modelLayers, int numLayers);
    void ReLU(Neuron**, int rows, int cols);
    void der_ReLU(Neuron** matrix, int rows, int cols);
    Neuron** matrixMultiply(Neuron** matrixOne, int rowsOne, int colsOne, Neuron** matrixTwo, int rowsTwo, int colsTwo)
    Neuron** forwardPropagate(Neuron** weightsMatrix, int weightsRows, int weightsCols, Neuron** inputMatrix, int inputsRows, int inputCols, Neuron* biasMatrix);
};


