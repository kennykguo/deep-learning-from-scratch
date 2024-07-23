#include "LinearLayer.h"
#include "Network.h"
#include "Neuron.h"
#include "ReLU.h"
#include "SoftmaxCrossEntropy.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <chrono>
#include <iomanip>
#include <memory>

using namespace std;

const int INPUT_SIZE = 784; // Number of pixels in each image
const int BATCH_SIZE = 16; // Batch size for processing

tuple<vector<vector<Neuron>>, vector<vector<vector<Neuron>>>> readCSV(const string& filename) {
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
        if (firstLine) {
            firstLine = false;
            continue; // Skip header line
        }

        stringstream ss(line);
        string cell;

        vector<Neuron> example;
        Neuron labelNeuron;

        // Read label
        if (!getline(ss, cell, ',')) {
            cerr << "Error reading label from line: " << line << endl;
            continue;
        }
        labelNeuron.value = stod(cell);
        currentLabelBatch.push_back(labelNeuron);

        // Read pixels
        for (int j = 0; j < INPUT_SIZE; ++j) {
            if (!getline(ss, cell, ',')) {
                cerr << "Error reading pixel " << j << " from line: " << line << endl;
                break;
            }
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

    // Add the last batch if it's not empty
    if (!currentBatch.empty()) {
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
    std::cout << "Starting program with CUDA acceleration:\n";

    // Define our data
    std::string filename = "train.csv";
    auto [labels, batches] = readCSV(filename);

    // Define model layers
    Network network;
    network.addLayer(std::make_unique<LinearLayer>(784, 256));
    network.addLayer(std::make_unique<ReLU>());
    network.addLayer(std::make_unique<LinearLayer>(256, 128));
    network.addLayer(std::make_unique<ReLU>());
    network.addLayer(std::make_unique<LinearLayer>(128, 10));
    network.addLayer(std::make_unique<SoftmaxCrossEntropy>(10));

    std::cout << "Labels size: " << labels.size() << "\n";
    std::cout << "Batches size: " << batches.size() << "\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    // Training loop
    for (size_t epoch = 0; epoch < 5; ++epoch) {  // 5 epochs as an example
        for (size_t i = 0; i < batches.size(); ++i) {
            auto output = network.forward(batches[i], {labels[i]});  // Wrap labels[i] in a vector
            network.backward();
            
            if (i % 100 == 0) {  // Print loss every 100 batches
                std::cout << "Epoch " << epoch << ", Batch " << i << ", Loss: " << network.getLoss() << std::endl;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Training completed. Total runtime: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}