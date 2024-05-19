#include <cstdio>
#include <Network.cpp>

using namespace std;


int main() {
    int modelLayers[] = {3, 2, 1}; // Example layer sizes
    int fan_in[] = {3, 2}; // Example fan_in for each layer
    int fan_out[] = {2, 1}; // Example fan_out for each layer
    int numLayers = sizeof(modelLayers) / sizeof(modelLayers[0]);
    Network myNetwork(modelLayers, numLayers, fan_in, fan_out);

    // Use myNetwork...

    return 0;
}
