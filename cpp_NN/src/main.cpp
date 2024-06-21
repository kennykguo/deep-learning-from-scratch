#include "LinearLayer.h"
#include "Network.h"
#include "Neuron.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>

using namespace std;

const int INPUT_SIZE = 784; // Number of pixels in each image
const int BATCH_SIZE = 16; // Batch size for processing

// Function to read CSV file and return a tuple of labels and batches
tuple< vector<vector<Neuron>>, vector<vector<vector<Neuron>>> > readCSV(const string& filename) {
    
    ifstream file(filename);

    vector<vector<Neuron>> labels;
    vector<vector<vector<Neuron>>> batches;

    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return make_tuple(labels, batches);
    }

    string line;
    bool firstLine = true;

    vector<Neuron> currentLabelBatch;
    vector<vector<Neuron>> currentBatch;

    while (getline(file, line)) {
        // Skips the first line
        if (firstLine) {
            firstLine = false;
            continue; // Skip header line
        }

        // Initializes a string stream with the current line for parsing
        stringstream ss(line);
        string cell;

        // One example and one label
        vector<Neuron> example;
        Neuron labelNeuron;

        // Read label
        getline(ss, cell, ',');
        labelNeuron.value = stod(cell);
        currentLabelBatch.push_back(labelNeuron);

        // Read pixels
        for (int j = 0; j < INPUT_SIZE && getline(ss, cell, ','); ++j) {
            Neuron pixelNeuron;
            pixelNeuron.value = stod(cell) / 255.0; // Normalize pixel value
            example.push_back(pixelNeuron);
        }

        currentBatch.push_back(example);

        // If batch is full, add it to batches and reset
        if (currentBatch.size() == BATCH_SIZE) {
            batches.push_back(currentBatch);
            labels.push_back(currentLabelBatch);
            currentBatch.clear();
            currentLabelBatch.clear();
        }
    }

    // Remove the last batch if it's not full
    if (currentBatch.size() != BATCH_SIZE) {
        currentBatch.clear();
        currentLabelBatch.clear();
    } else {
        batches.push_back(currentBatch);
        labels.push_back(currentLabelBatch);
    }

    file.close();
    return make_tuple(labels, batches);
}

void printExample(const vector<Neuron>& label, const vector<vector<Neuron>>& example) {
    // Print the label
    cout << "Label: " << label[0].value << endl;

    // Print the pixels in a 28x28 grid
    cout << "Pixels:" << endl;
    for (size_t i = 0; i < example.size(); ++i) {
        for (size_t j = 0; j < example[i].size(); ++j) {
            if (j > 0 && j % 28 == 0) {
                cout << endl;
            }
            cout << example[i][j].value << " ";
        }
        cout << endl;
    }
}




int main() {

    cout << "Starting program:\n";

    // Define our data
    string filename = "train.csv";
    tuple<vector<vector<Neuron>>, vector<vector<vector<Neuron>>>> result = readCSV(filename);
    

    // Define model layers manually
    Network network;
    network.networkLayers.push_back(LinearLayer(network, INPUT_SIZE, 128)); // (784, 128)
    network.networkLayers.push_back(LinearLayer(network, 128, 64)); // (128, 64)
    network.networkLayers.push_back(LinearLayer(network, 64, 10)); // (64, 10)

    // Example usage: print the first label and batch
    // Check if batch and labels are not empty
    vector<vector<Neuron>> labels = get<0>(result);
    vector<vector<vector<Neuron>>> batches = get<1>(result);
    if (!labels.empty() && !batches.empty()) {
        // Print the first label and batch for verification
        // printExample(labels[0], batches[0]);

        // Forward pass for the first batch
        vector<vector<Neuron>> inputBatch = batches[0];

        // Forward through the first layer
        vector<vector<Neuron>> layer1Output = network.networkLayers[0].forward(inputBatch);

        // Forward through the second layer
        vector<vector<Neuron>> layer2Output = network.networkLayers[1].forward(layer1Output);

        // Forward through the third layer
        vector<vector<Neuron>> finalOutput = network.networkLayers[2].forward(layer2Output);

        // Print the output of the last layer (should be of shape [BATCH_SIZE, 10])
        // cout << "Output:" << endl;
        // for (const auto& row : finalOutput) {
        //     for (const auto& neuron : row) {
        //         cout << neuron.value << " ";
        //     }
        //     cout << endl;
        // }
    }

    cout << "Done" << endl;
    return 0;
}
