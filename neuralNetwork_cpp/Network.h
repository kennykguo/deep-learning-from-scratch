#include <linearLayer.cpp>
#include <cstdio>
using namespace std;

class Network
{
public:
    Layer* networkLayers;
    Network(int* numLayers);
    void Network::ReLU(Neuron**, int rows, int cols);
    void der_ReLU(Neuron** matrix, int rows, int cols);
    Neuron** matrixMultiply(Neuron** matrixOne, int rowsOne, int colsOne, Neuron** matrixTwo, int rowsTwo, int colsTwo);
};


