// Training.h
#ifndef TRAINING_H
#define TRAINING_H

#include <vector>
#include <string>
#include <fstream>

using namespace std;

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
    ifstream m_trainingDataFile;
};

#endif // TRAINING_H
