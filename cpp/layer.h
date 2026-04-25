#ifndef _LAYER_H_
#define _LAYER_H_

#define HIDDEN_SIZE 256

void intermediateLayer(float layerInput[HIDDEN_SIZE],
    float layerOutput[HIDDEN_SIZE],
    const float weights[HIDDEN_SIZE][HIDDEN_SIZE],
    const float bias[HIDDEN_SIZE]);

#endif