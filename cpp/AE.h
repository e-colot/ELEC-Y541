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

void AE(
    int imINPUT[MAX_HEIGHT][MAX_WIDTH], 
    int imOUTPUT[MAX_HEIGHT][MAX_WIDTH],
    const float enc1[HIDDEN_SIZE][INPUT_SIZE],
    const float enc1_bias[HIDDEN_SIZE],
    const float enc2[HIDDEN_SIZE][HIDDEN_SIZE],
    const float enc2_bias[HIDDEN_SIZE],
    const float enc3[HIDDEN_SIZE][HIDDEN_SIZE],
    const float enc3_bias[HIDDEN_SIZE],
    const float enc4[CODE_SIZE][HIDDEN_SIZE],
    const float enc4_bias[CODE_SIZE],
    const float dec1[HIDDEN_SIZE][CODE_SIZE],
    const float dec1_bias[HIDDEN_SIZE],
    const float dec2[HIDDEN_SIZE][HIDDEN_SIZE],
    const float dec2_bias[HIDDEN_SIZE],
    const float dec3[HIDDEN_SIZE][HIDDEN_SIZE],
    const float dec3_bias[HIDDEN_SIZE],
    const float dec4[INPUT_SIZE][HIDDEN_SIZE],
    const float dec4_bias[INPUT_SIZE]
);

void inputLayer(float layerInput[INPUT_SIZE],
            float layerOutput[HIDDEN_SIZE],
            const float weights[HIDDEN_SIZE][INPUT_SIZE],
            const float bias[HIDDEN_SIZE]);
void intermediateLayer(float layerInput[HIDDEN_SIZE],
            float layerOutput[HIDDEN_SIZE],
            const float weights[HIDDEN_SIZE][HIDDEN_SIZE],
            const float bias[HIDDEN_SIZE]);
void endEncryptionLayer(float layerInput[HIDDEN_SIZE],
            float layerOutput[CODE_SIZE],
            const float weights[CODE_SIZE][HIDDEN_SIZE],
            const float bias[CODE_SIZE]);
void startDecryptionLayer(float layerInput[CODE_SIZE],
            float layerOutput[HIDDEN_SIZE],
            const float weights[HIDDEN_SIZE][CODE_SIZE],
            const float bias[HIDDEN_SIZE]);
void outputLayer(float layerInput[HIDDEN_SIZE],
            float layerOutput[INPUT_SIZE],
            const float weights[INPUT_SIZE][HIDDEN_SIZE],
            const float bias[INPUT_SIZE]);

void flatten(int imINPUT[MAX_HEIGHT][MAX_WIDTH], float layerInput[INPUT_SIZE]);
void unflatten(float layerOutput[INPUT_SIZE], int imOUTPUT[MAX_HEIGHT][MAX_WIDTH]);

#endif
