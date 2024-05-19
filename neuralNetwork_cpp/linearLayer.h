#include <cstdio>
#include <Neuron.cpp>
#include <Network.h>
using namespace std;

class Layer
{
public:
    // Each layer will have attributes which include a matrix of output activations, a weights matrix, and a bias matrix
    int fan_in;
    int fan_out;
    Neuron** weightsMatrix;
    Neuron* biasMatrix;
    Neuron** outputActivations;
    // Default constuctor
    Layer(int fan_in, int fan_out);
};