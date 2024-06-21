#include "LinearLayer.h"
#include "Network.h"
#include "Neuron.h"
#include <iostream>
#include <vector>

#include <fstream>
#include <sstream>
#include <string>

using namespace std;

const int INPUT_SIZE = 784; // Number of pixels in each image
const int BATCH_SIZE = 32; // Batch size for processing

// Function to read CSV file and return a vector of vector of Neurons
vector<vector<vector<Neuron>>> readCSV(const string& filename) {

    // Opens the file
    ifstream file(filename);

    vector<vector<vector<Neuron>>> batches;

    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return batches;
    }


    string line;
    bool firstLine = true;

    while (getline(file, line)) {
        // Skips the first line
        if (firstLine) {
            firstLine = false;
            continue; // Skip header line
        }

        // Initializes a string stream with the current line for parsing
        stringstream ss(line);
        // Initalizes empty vector for a single batch
        vector<vector<Neuron>> batch;
        // 
        string cell;

        // Read label and pixels
        for (int i = 0; i < BATCH_SIZE && getline(ss, cell, ','); ++i) {
            vector<Neuron> example;
            // Neuron object for the label
            Neuron labelNeuron;
            labelNeuron.value = stod(cell);
            example.push_back(labelNeuron);

            // Splits lines by ,
            for (int j = 0; j < INPUT_SIZE && getline(ss, cell, ','); ++j) {
                Neuron pixelNeuron;
                pixelNeuron.value = stod(cell) / 255.0; // Normalize pixel value
                example.push_back(pixelNeuron);
            }
            // Adds example to batch
            batch.push_back(example);
        }
        // Adds batch to batches
        batches.push_back(batch);
    }

    file.close();
    return batches;
}


void printExample(const vector<vector<Neuron>>& example) {
    // First neuron in the example is assumed to be the label
    cout << "Label: " << example[0][0].value << endl;

    // Remaining neurons are assumed to be pixels
    cout << "Pixels: ";
    for (size_t i = 1; i < example.size(); ++i) {
        cout << example[i][0].value << " ";
    }
    cout << endl;
}



int main() {
    
    cout << "Starting program\n";

    // Define model layers
    vector<int> modelLayers = {INPUT_SIZE, 128, 64, 10}; // Example architecture

    // Initalize the network
    Network network(modelLayers);

    // Initialize the input data
    vector<vector<Neuron>> input(3, vector<Neuron>(3));
    network.networkLayers[0].outputActivations = input;


    // Read batches from CSV
    vector<vector<vector<Neuron>>> batches = readCSV("train.csv");

    // // Forward propagation through the network
    // for (int layerNum = 0; layerNum < network.numLayers - 1; layerNum++) {
    //     network.networkLayers[layerNum + 1].outputActivations = 
    //         network.networkLayers[layerNum].forward(network.networkLayers[layerNum].outputActivations);
    // }

    // return 0;

    const string filename = "train.csv";
    vector<vector<vector<Neuron>>> batches = readCSV(filename);

    // Assuming you want to visualize the first example in the first batch
    printExample(batches[0][0]);
    
}
