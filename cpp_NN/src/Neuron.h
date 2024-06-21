#pragma once

#include <random>

class Neuron {
public:
    Neuron();
    double value;
    double gradient;
    static double randomValue() {
        return rand() / double(RAND_MAX); 
    }
};
