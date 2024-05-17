// Library for dynamic arrays
#include <vector>
// Library for input and output streams
#include <iostream>

// Brings the symbols in the 'std' namespace into the global namespace
using namespace std;

// Neuron class
class Neuron{};

// Layer is a vector of Neuron objects
typedef vector<Neuron> Layer;


class Net
{

public:
    // Constructor
    Net(const vector<unsigned> &topology);
    // Methods
    void feedForward(const vector<double> &inputVals){};
    void backProp(const vector<double> &targetVals){};
    void getResults(const vector<double> &resultVals) const {};

private:
    // Vector of layer objects
    vector<Layer> m_layers;
    //m_layers[layerNum][neuronNum]
};


// Net constructor
Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        m_layers.push_back(Layer());
        // Plus and equal for the "bias neuron"
        for (unsigned neuronNum = 0; neuronNum<=topology[layerNum]; neuronNum++)
        {
            m_layers.back().push_back(Neuron());
            cout <<"Made a Neuron!" << endl;
        }
    }
}


int main()
{
    vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    Net myNet(topology);

    vector<double> inputVals;
    myNet.feedForward(inputVals);

    vector<double> targetVals;
    myNet.backProp(targetVals);

    vector<double> resultVals;
    myNet.getResults(resultVals);

}