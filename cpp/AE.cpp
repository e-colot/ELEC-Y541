// AE.cpp

#include "AE.h"
#include <algorithm>

using namespace std;

// Top function
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
  float dec4_bias[INPUT_SIZE]) {
      // enc are weights of the encoding layer
      // dec are weights of the decoding layer

      float flattenedInput[INPUT_SIZE];
      FlattenOut:for(int row = 0; row < MAX_HEIGHT; row++) {
        FlattenIn:for(int col = 0; col < MAX_WIDTH; col++) {
          // Normalize input: (pixel/255 - mean) / std
          float normalized = (static_cast<float>(imINPUT[row][col]) / 255.0f - AE_NORM_MEAN) / AE_NORM_STD;
          flattenedInput[row*MAX_WIDTH + col] = normalized;
        }
      }

      // encoding
      float outEnc1[INPUT_SIZE];
      Layer(flattenedInput, outEnc1, enc1, enc1_bias, INPUT_SIZE, HIDDEN_SIZE, true);
      float outEnc2[INPUT_SIZE];
      Layer(outEnc1, outEnc2, enc2, enc2_bias, HIDDEN_SIZE, HIDDEN_SIZE, true);
      float outEnc3[INPUT_SIZE];
      Layer(outEnc2, outEnc3, enc3, enc3_bias, HIDDEN_SIZE, HIDDEN_SIZE, true);
      float outEnc4[INPUT_SIZE];
      Layer(outEnc3, outEnc4, enc4, enc4_bias, HIDDEN_SIZE, CODE_SIZE, true);
      // decoding
      float outDec1[INPUT_SIZE];
      Layer(outEnc4, outDec1, dec1, dec1_bias, CODE_SIZE, HIDDEN_SIZE, true);
      float outDec2[INPUT_SIZE];
      Layer(outDec1, outDec2, dec2, dec2_bias, HIDDEN_SIZE, HIDDEN_SIZE, true);
      float outDec3[INPUT_SIZE];
      Layer(outDec2, outDec3, dec3, dec3_bias, HIDDEN_SIZE, HIDDEN_SIZE, true);
      float outDec4[INPUT_SIZE];
      Layer(outDec3, outDec4, dec4, dec4_bias, HIDDEN_SIZE, INPUT_SIZE, false);

      shapingOut:for(int row = 0; row < MAX_HEIGHT; row++) {
        shapingIn:for(int col = 0; col < MAX_WIDTH; col++) {
          // Denormalize and scale to [0, 255]
          float denorm = AE_DENORM(outDec4[row*MAX_WIDTH + col]);
          imOUTPUT[row][col] = AE_SCALE_OUTPUT(denorm);
        }
      }


}

void Layer(float layerInput[INPUT_SIZE],
      float layerOutput[INPUT_SIZE],
      float weights[INPUT_SIZE*HIDDEN_SIZE],
      float bias[INPUT_SIZE],
      int in_size, int out_size, bool is_relu) {

    // Matrix multiplication with bias addition
    float intermediateLayer[INPUT_SIZE];

    L0:for(int row = 0; row < out_size; row++) {
      float tmp = bias[row];  // Start with bias
      L1:for(int col = 0; col < in_size; col++) {
        tmp = tmp + weights[row*in_size + col] * layerInput[col];
      }
      // ReLu
      if (is_relu) {
        intermediateLayer[row] = max(0.0f, tmp);
      }
      else {
        intermediateLayer[row] = tmp;
      }
    }
    
    // Copy to output
    for(int i = 0; i < out_size; i++) {
      layerOutput[i] = intermediateLayer[i];
    }
  }

