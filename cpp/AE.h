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

// Denormalization macro: unnormalized = normalized * std + mean
#define AE_DENORM(x) ((x) * AE_NORM_STD + AE_NORM_MEAN)

// Output scaling: multiply by 255 before casting to uint8
#define AE_SCALE_OUTPUT(x) ((int)((x) * 255.0f) & 0xFF)

void AE(int imINPUT[MAX_HEIGHT][MAX_WIDTH], 
    int imOUTPUT[MAX_HEIGHT][MAX_WIDTH], 
    float enc1[INPUT_SIZE*HIDDEN_SIZE], 
    float enc2[INPUT_SIZE*HIDDEN_SIZE], 
    float enc3[INPUT_SIZE*HIDDEN_SIZE], 
    float enc4[INPUT_SIZE*HIDDEN_SIZE],
    float enc1_bias[INPUT_SIZE],
    float enc2_bias[INPUT_SIZE],
    float enc3_bias[INPUT_SIZE],
    float enc4_bias[INPUT_SIZE],
    float dec1[INPUT_SIZE*HIDDEN_SIZE], 
    float dec2[INPUT_SIZE*HIDDEN_SIZE], 
    float dec3[INPUT_SIZE*HIDDEN_SIZE], 
    float dec4[INPUT_SIZE*HIDDEN_SIZE],
    float dec1_bias[INPUT_SIZE],
    float dec2_bias[INPUT_SIZE],
    float dec3_bias[INPUT_SIZE],
    float dec4_bias[INPUT_SIZE]);

// Keeping input size as dimensions for both input and output as it is the largest
void Layer(float layerInput[INPUT_SIZE],
            float layerOutput[INPUT_SIZE],
            float weights[INPUT_SIZE*HIDDEN_SIZE],
            float bias[HIDDEN_SIZE],
            int in_size, int out_size, bool is_relu);

#endif
