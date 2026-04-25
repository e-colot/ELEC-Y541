// AE_stream.cpp

#include "AE.h"
// #include "constant.h"
#include <algorithm>
#include <cmath>

using namespace std;

// Top function
void AE(
  int imINPUT[MAX_HEIGHT][MAX_WIDTH], 
  int imOUTPUT[MAX_HEIGHT][MAX_WIDTH],
  // Explicitly pass weights and biases to satisfy Dataflow requirements
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
) {

      float flattenedInput[INPUT_SIZE];
      flatten(imINPUT, flattenedInput);

      // encoding
      float h1[HIDDEN_SIZE];
      inputLayer(flattenedInput, h1, enc1, enc1_bias);
      float h2[HIDDEN_SIZE];
      intermediateLayer(h1, h2, enc2, enc2_bias);
      float h3[HIDDEN_SIZE];
      intermediateLayer(h2, h3, enc3, enc3_bias);
      float code[CODE_SIZE];
      endEncryptionLayer(h3, code, enc4, enc4_bias);

      // decoding
      float h5[HIDDEN_SIZE];
      startDecryptionLayer(code, h5, dec1, dec1_bias);
      float h6[HIDDEN_SIZE];
      intermediateLayer(h5, h6, dec2, dec2_bias);
      float h7[HIDDEN_SIZE];
      intermediateLayer(h6, h7, dec3, dec3_bias);
      float output[INPUT_SIZE];
      outputLayer(h7, output, dec4, dec4_bias);

      unflatten(output, imOUTPUT);
}

void flatten(int imINPUT[MAX_HEIGHT][MAX_WIDTH], float layerInput[INPUT_SIZE]) {
    FlattenOut:for(int row = 0; row < MAX_HEIGHT; row++) {
        FlattenIn:for(int col = 0; col < MAX_WIDTH; col++) {
            float normalized = (static_cast<float>(imINPUT[row][col]) / 255.0f - AE_NORM_MEAN) / AE_NORM_STD;
            layerInput[row*MAX_WIDTH + col] = normalized;
        }
    }
}

void unflatten(float layerOutput[INPUT_SIZE], int imOUTPUT[MAX_HEIGHT][MAX_WIDTH]) {
    shapingOut:for(int row = 0; row < MAX_HEIGHT; row++) {
        shapingIn:for(int col = 0; col < MAX_WIDTH; col++) {
            float scaled = layerOutput[row*MAX_WIDTH + col] * 255.0f;
            int rounded = static_cast<int>(scaled+0.5f);
            if (rounded > 255) rounded = 255;
            else if (rounded < 0) rounded = 0;
            imOUTPUT[row][col] = rounded;
        }
    }
}

void inputLayer(float layerInput[INPUT_SIZE],
            float layerOutput[HIDDEN_SIZE],
            const float weights[HIDDEN_SIZE][INPUT_SIZE],
        const float bias[HIDDEN_SIZE]) {

    float input_buffer[INPUT_SIZE];
    float tmp; 

    L0:for(int row = 0; row < HIDDEN_SIZE; row++) {

        tmp = bias[row]; // Start with bias

        L1:for(int col = 0; col < INPUT_SIZE; col++) {

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

void intermediateLayer(float layerInput[HIDDEN_SIZE],
            float layerOutput[HIDDEN_SIZE],
            const float weights[HIDDEN_SIZE][HIDDEN_SIZE],
            const float bias[HIDDEN_SIZE]) {

    float input_buffer[HIDDEN_SIZE];
    float tmp; 

    L0:for(int row = 0; row < HIDDEN_SIZE; row++) {

        tmp = bias[row]; // Start with bias
      
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

void endEncryptionLayer(float layerInput[HIDDEN_SIZE],
            float layerOutput[CODE_SIZE],
            const float weights[CODE_SIZE][HIDDEN_SIZE],
            const float bias[CODE_SIZE]) { 

    float input_buffer[HIDDEN_SIZE];
    float tmp; 

    L0:for(int row = 0; row < CODE_SIZE; row++) {

        tmp = bias[row]; // Start with bias
      
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

void startDecryptionLayer(float layerInput[CODE_SIZE],
            float layerOutput[HIDDEN_SIZE],
            const float weights[HIDDEN_SIZE][CODE_SIZE],
            const float bias[HIDDEN_SIZE]) { 

    float input_buffer[CODE_SIZE];
    float tmp; 

    L0:for(int row = 0; row < HIDDEN_SIZE; row++) {

        tmp = bias[row]; // Start with bias
      
      L1:for(int col = 0; col < CODE_SIZE; col++) {

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

void outputLayer(float layerInput[HIDDEN_SIZE],
            float layerOutput[INPUT_SIZE],
            const float weights[INPUT_SIZE][HIDDEN_SIZE],
            const float bias[INPUT_SIZE]) { 

    float input_buffer[HIDDEN_SIZE];
    float tmp; 

    L0:for(int row = 0; row < INPUT_SIZE; row++) {

        tmp = bias[row]; // Start with bias
      
      L1:for(int col = 0; col < HIDDEN_SIZE; col++) {

        // Fill input buffer only at first iteration
        if (row == 0) {
          input_buffer[col] = layerInput[col];
        }

        tmp = tmp + weights[row][col] * input_buffer[col];
      }
      // no ReLu
      layerOutput[row] = tmp;
    }
  }


