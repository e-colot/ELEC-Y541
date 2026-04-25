# include "layer.h"
#include <algorithm>

using namespace std;

void intermediateLayer(float layerInput[HIDDEN_SIZE],
    float layerOutput[HIDDEN_SIZE],
    const float weights[HIDDEN_SIZE][HIDDEN_SIZE],
    const float bias[HIDDEN_SIZE]) {

        
    L0_intermediate:for(int row = 0; row < HIDDEN_SIZE; row++) {

        L1_intermediate:for(int col = 0; col < HIDDEN_SIZE; col++) {

            layerOutput[row] += weights[row][col] * layerInput[col];
        }
        // ReLu
        layerOutput[row] = max(0.0f, layerOutput[row] + bias[row]);
    }
}
