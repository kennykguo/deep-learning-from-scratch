#include "Neuron.h"
#include <cstdlib> // Include for rand()

Neuron::Neuron() {
    // Initialize a random value when the object is instantiated
    value = randomValue();
    // Initialize the gradient to zero for now
    gradient = 0;
}
