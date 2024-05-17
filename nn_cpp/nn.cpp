// Library for dynamic arrays
#include <vector>
// Library for input and output streams
#include <iostream>
#include <cassert>
#include <cmath>

// Brings the symbols in the 'std' namespace into the global namespace
using namespace std;


struct Connection
{
    double weight;
    double deltaWeight;
}

// Neuron class
class Neuron;

// Layer is a vector of Neuron objects
typedef vector<Neuron> Layer;


class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val){ m_outputVal = val; }
    double getOutputVal(void) const{ return m_outputVal; }
    void feedForward(const Layer &prevLayer);

private:
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void){ return rand() / double(RAND_MAX); }
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
};

double Neuron::transferFunction(double x){
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x);
{
    return 1.0 - x * x;
}

void Neuron:: feedForward(const Layer &prevLayer)
{
    double sum = 0.0
    for (unsigned n = 0; n< prevLayer.size(); n++)
    {
        sum +=prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex]..wieght;
    }
}



Neuron::Neuron (unsigned numOutputs, unsigned MyIndex)
{
    for (unsigned c = 0; c < numOutputs; c++)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = MyIndex;
}



class Net
{
// Members declared under the 'public' section are accessible from outside the class
// Functions, variables, that the users of the class can access and use (main function)
public:
    // Call the Net constructor
    Net(const vector<unsigned> &topology);
    // Methods
    void feedForward(const vector<double> &inputVals){

    };
    void backProp(const vector<double> &targetVals){

    };
    void getResults(const vector<double> &resultVals) const {

    };
    


// Members declared under the 'private' section are only accessible from within the class itself
// They store the internal implementation details - can only be accessed by methods within the class itself
private:
    // Vector of layer objects
    vector<Layer> m_layers;
    //m_layers[layerNum][neuronNum]
};



void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals,size() = m_layers[0].size() - 1);
    for (unsigned i = 0; i<inputVals,size(); i++)
    {
        m_layers[0][i].setOutputVal(inputVals[i]); 
    }
    // Forward propogate
    for (unsigned layerNum = 1; layerNum <m_layers.size(); layer++){
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n<m_layers[layer_num].size() - 1; n++)
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
    m_outputVal = transferFunction(sum);
}



// Net constructor (like the __init__ function in Python)
// Net:: specifies that this constructor belongs to the Net class
Net::Net(const vector<unsigned> &topology)
{
    // Gets the length of the topology vector
    unsigned numLayers = topology.size();
    // For each layer, instantiate a Layer class. For each Layer class, add the respective number of Neuron classes
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() -1 ? 0 : topology[layerNum + 1];

        // Plus and equal for the "bias neuron"
        for (unsigned neuronNum = 0; neuronNum<=topology[layerNum]; neuronNum++)
        {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            // endl is a manipulator for output streams and inserts a newline character, flusing out the new stream
            cout <<"Made a Neuron!" << endl;
        }
    }
}


int main()
{
    vector<unsigned> topology;
    // Adds an element to the end of the vector
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    // Create a Net object, and pass in the topology vector
    Net myNet(topology);

    // Call the functions
    vector<double> inputVals;
    myNet.feedForward(inputVals);

    vector<double> targetVals;
    myNet.backProp(targetVals);

    vector<double> resultVals;
    myNet.getResults(resultVals);

}