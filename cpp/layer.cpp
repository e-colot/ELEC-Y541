# include "layer.h"
#include <algorithm>

using namespace std;

void intermediateLayer(float layerInput[HIDDEN_SIZE],
    float layerOutput[HIDDEN_SIZE],
    const float weights[HIDDEN_SIZE][HIDDEN_SIZE],
    const float bias[HIDDEN_SIZE]) {

    float input_buffer[HIDDEN_SIZE];
    float tmp;
        
    L0:for(int row = 0; row < HIDDEN_SIZE; row++) {

        tmp = bias[row];  // Start with bias

        L1:for(int col = 0; col < HIDDEN_SIZE; col++) {

            // Fill input buffer only at first iteration
            if (row == 0) {
            input_buffer[col] = layerInput[col];
            }

            tmp = tmp + weights[row][col] * input_buffer[col];
        }
        // ReLu
        layerOutput[row] = max(0.0f, tmp);
    }
}
