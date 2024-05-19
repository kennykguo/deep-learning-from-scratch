#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

// Silly class to read training data from a text file -- Replace This.
// Replace class TrainingData with whatever you need to get input data into the
// program, e.g., connect to a database, or take a stream of data from stdin, or
// from a file specified by a command line argument, etc.
// TODO
// Figure out what static means
// Start with Net class
// Figure out what is being passed as a pointer, and what is passed by value
// Figure out now namespaces are working in the file

class TrainingData
{
public:
    TrainingData(const string filename);
    // Special function
    bool isEof(void) {
        return m_trainingDataFile.eof(); 
    }
    void getTopology(vector<unsigned> &topology);
    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    // ifstream is a class that allows for file reading
    ifstream m_trainingDataFile;
};

// Takes in filename, and opens the filename using an ifstream object
TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

// Reads the first line of training data (input, hidden, output)
// void TrainingData::getTopology(vector<unsigned> &topology)
// {
//     string line;
//     string label;

//     getline(m_trainingDataFile, line);
//     stringstream ss(line);
//     ss >> label;
//     if (this->isEof() || label.compare("topology:") != 0) {
//         abort();
//     }

//     while (!ss.eof()) {
//         unsigned n;
//         ss >> n;
//         topology.push_back(n);
//     }

//     return;
// }

// Reads the next line of the training data and extracts input values
// Values are expected to be preceded by the label 'in:'
unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

// Reads the next line of the training data and extracts input values
// Values are expected to be preceded by the label 'in:'
unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}


// A single connection holds its weight, and its respective gradient
struct Connection
{
    double weight;
    double deltaWeight;
};


class Neuron;

// typedef vector<Neuron> Layer;

// ****************** class Neuron ******************
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
    // Static means that you can call these functions without having to create a Neuron class
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
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

// Neuron constructor
Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{

    for (unsigned c = 0; c < numOutputs; ++c) {
        // Creates a new connection object and appends it the attribute 'm_outputWeights'
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

// Takes in a pointer to a vector of the previous layer
void Neuron::updateInputWeights(Layer &prevLayer)
{
    // For each neuron
    for (unsigned n = 0; n < prevLayer.size(); n++) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                eta
                * neuron.getOutputVal()
                * m_gradient
                // Also add momentum = a fraction of the previous delta weight;
                + alpha
                * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

// Sums the derivatives of the next layer
double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}

// To feedforward to the current neuron, we sum all the contributions of the previous layer's outputs * the outputWeights that are going into the index of the neuron
// This function will take in a pointer to the previous layer. It will forward all the neurons of the previous layer into the neuron, multiplying by the respective m_myIndex's weights going into that neuron. It will then apply the activation function
void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    // Sum the previous layer's activations
    // Include the bias node from the previous layer.
    for (unsigned n = 0; n < prevLayer.size(); n++) {
        // prevLayer[n] gets the Neuron
        // .getOutputVal() gets the output value
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}


//  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //


// Class Net
class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }
// private:
    vector<Layer> m_layers; 
    // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};


double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over



Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size(); // 3
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // We have a new layer, now fill it with neurons, and
        // add a bias neuron in each layer.
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a Neuron" << endl;
        }

        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        // ????
        m_layers.back().back().setOutputVal(1.0);
        cout << "Updated bias Neuron" << endl;
    }
}

void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

// Passed the address of the targetVals vector
void Net::backProp(const vector<double> &targetVals)
{
    // Assign a pointer to the last Layer
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    // 
    for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error); // RMS

    // Implement a recent average measurement

    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate hidden layer gradients

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // Update connection weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

// This function is called first in main when the user has intialized all the respective vectors correctly
void Net::feedForward(const vector<double> &inputVals)
{
    // Makes sure the number of input neurons matches the size of the inputVals vector
    assert(inputVals.size() == m_layers[0].size() - 1);

    // Loop over all of the input values and assign m_outputValue in each neuron to the respective value
    for (unsigned i = 0; i < inputVals.size(); i++) {
        // Initalize the input vector to the corresponding neuron in first layer
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagate each layer after, until the end
    // This will do it for layer 1 and 2
    for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
        // Create a pointer to the last layer
        Layer &prevLayer = m_layers[layerNum - 1];
        // Feed forward the previous layer
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

using namespace std;

int main()
{
    TrainingData trainData("training.txt");

    vector<unsigned> topology;

    topology.push_back(3);

    topology.push_back(4);

    topology.push_back(1);

    Net myNet(topology);

    // Create a vector of inputs, targets, and results
    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) {
        trainingPass++;
        cout << endl << "Pass " << trainingPass;

        // Get input data, if error, then break out of training loop
        if (trainData.getNextInputs(inputVals) != topology[0]) { // 3
            cout << "Input data is not in the correct format";
            break;
        }
        showVectorVals(": Inputs:", inputVals);

        //Feed forward input values
        myNet.feedForward(inputVals);

        // Get the neural network's output results
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Get output results from data
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);

        // Output values = # of neurons in last layer
        assert(targetVals.size() == topology.back());

        // Backpropogate the network and update the weights
        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
    }
    cout << endl << "Done" << endl;
}

0