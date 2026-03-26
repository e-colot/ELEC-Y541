// AE.h
#ifndef _AE_H_
#define _AE_H_

#define MAX_WIDTH 28
#define MAX_HEIGHT 28

#define INPUT_SIZE 784 // 28*28
#define HIDDEN_SIZE 256
#define CODE_SIZE 128

#define AE_NORM_MEAN  0.1307f
#define AE_NORM_STD   0.3081f

void AE(int imINPUT[MAX_HEIGHT][MAX_WIDTH], 
    int imOUTPUT[MAX_HEIGHT][MAX_WIDTH]);

// Keeping input size as dimensions for both input and output as it is the largest
void inputLayer(float layerInput[INPUT_SIZE],
            float layerOutput[HIDDEN_SIZE],
            float weights[HIDDEN_SIZE][INPUT_SIZE],
            float bias[HIDDEN_SIZE]);
void intermediateLayer(float layerInput[HIDDEN_SIZE],
            float layerOutput[HIDDEN_SIZE],
            float weights[HIDDEN_SIZE][HIDDEN_SIZE],
            float bias[HIDDEN_SIZE]);
void endEncryptionLayer(float layerInput[HIDDEN_SIZE],
            float layerOutput[CODE_SIZE],
            float weights[CODE_SIZE][HIDDEN_SIZE],
            float bias[CODE_SIZE]);
void startDecryptionLayer(float layerInput[CODE_SIZE],
            float layerOutput[HIDDEN_SIZE],
            float weights[HIDDEN_SIZE][CODE_SIZE],
            float bias[HIDDEN_SIZE]);
void outputLayer(float layerInput[HIDDEN_SIZE],
            float layerOutput[INPUT_SIZE],
            float weights[INPUT_SIZE][HIDDEN_SIZE],
            float bias[INPUT_SIZE]);

#endif
