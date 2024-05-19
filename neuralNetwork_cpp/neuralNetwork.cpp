// main.cpp
#include <iostream>
#include <vector>
#include "Network.h"
#include "Training.h"

using namespace std;

void showVectorVals(string label, vector<double> &v);

int main() {

    TrainingData trainData("/tmp/trainingData.txt");

    vector<unsigned> topology;

    topology.push_back(3);

    topology.push_back(4);

    topology.push_back(5);

    Net myNet(topology);

    // // Simulate the training loop (replace with actual training data handling)
    // while (trainingPass < 100) { // Just a dummy loop for illustration
    //     ++trainingPass;
    //     cout << endl << "Pass " << trainingPass;

    //     // Simulate getting new input data:
    //     inputVals = {1.0, 0.5, -1.5};
    //     showVectorVals("Inputs:", inputVals);
    //     myNet.feedForward(inputVals);

    //     // Collect the net's actual output results:
    //     myNet.getResults(resultVals);
    //     showVectorVals("Outputs:", resultVals);

    //     // Simulate getting target output values:
    //     targetVals = {0.0, 1.0, 0.0, 1.0, 0.0};
    //     showVectorVals("Targets:", targetVals);

    //     myNet.backProp(targetVals);

    //     // Report the recent average error:
    //     cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
    // }

    cout << endl << "Done" << endl;
    return 0;
}

void showVectorVals(string label, vector<double> &v) {
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }
    cout << endl;
}
