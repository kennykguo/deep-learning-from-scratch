#include <vector>

using namespace std;

typedef vector<Neuron> Layer;

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron
{
public:
    // Neuron constructor that takes in the number of outputs and its index in the layer
    Neuron(unsigned numOutputs, unsigned myIndex);
    // Set neuron's output value (since it is private)
    void setOutputVal(double val) {
        m_outputVal = val; 
    }
    // Get neuron's output value (since it is private)
    double getOutputVal(void) const {
        return m_outputVal; 
    }
    // Calculates output values, given a pointer from the previous layer
    void feedForward(const Layer &prevLayer);
    // Calculates the output gradients, given the targetVal 
    void calcOutputGradients(double targetVal);
    // Calculates gradients for the next layer
    void calcHiddenGradients(const Layer &nextLayer);
    // Updates input weights of the previous layers
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;   
    static double alpha;
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    // What is static?
    // Returns a random weight
    static double randomWeight(void) {
        return rand() / double(RAND_MAX); 
    }
    double sumDOW(const Layer &nextLayer) const;
    // Each neuron will have a variable that:
    // Stores its output value
    // Stores its output weights
    // Stores its index in the layer
    // Stores its individual neuron gradient
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;
