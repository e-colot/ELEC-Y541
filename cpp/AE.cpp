// AE.cpp

#include "AE.h"
#include "constant.h"
#include <algorithm>
#include <cmath>

using namespace std;

static int scale_output_like_python(float x) {
  int v = static_cast<int>(std::lround(x * 255.0f));
  if (v < 0) return 0;
  if (v > 255) return 255;
  return v;
}

// Top function
void AE(int imINPUT[MAX_HEIGHT][MAX_WIDTH], 
  int imOUTPUT[MAX_HEIGHT][MAX_WIDTH]) {
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
      float output[CODE_SIZE];
      outputLayer(h7, output, dec4, dec4_bias);


      shapingOut:for(int row = 0; row < MAX_HEIGHT; row++) {
        shapingIn:for(int col = 0; col < MAX_WIDTH; col++) {
          // Match Python/OpenCV conversion: round then saturate to uint8 range.
          imOUTPUT[row][col] = scale_output_like_python(output[row*MAX_WIDTH + col]);
        }
      }


}

void inputLayer(float layerInput[INPUT_SIZE],
            float layerOutput[HIDDEN_SIZE],
            float weights[HIDDEN_SIZE][INPUT_SIZE],
            float bias[HIDDEN_SIZE]) {
    L0:for(int row = 0; row < HIDDEN_SIZE; row++) {
      float tmp = bias[row];  // Start with bias
      L1:for(int col = 0; col < INPUT_SIZE; col++) {
        tmp = tmp + weights[row][col] * layerInput[col];
      }
      // ReLu
      layerOutput[row] = max(0.0f, tmp);
    }
  }

void intermediateLayer(float layerInput[HIDDEN_SIZE],
            float layerOutput[HIDDEN_SIZE],
            float weights[HIDDEN_SIZE][HIDDEN_SIZE],
            float bias[HIDDEN_SIZE]) {
    L0:for(int row = 0; row < HIDDEN_SIZE; row++) {
      float tmp = bias[row];  // Start with bias
      L1:for(int col = 0; col < HIDDEN_SIZE; col++) {
        tmp = tmp + weights[row][col] * layerInput[col];
      }
      // ReLu
      layerOutput[row] = max(0.0f, tmp);
    }
  }

void endEncryptionLayer(float layerInput[HIDDEN_SIZE],
            float layerOutput[CODE_SIZE],
            float weights[CODE_SIZE][HIDDEN_SIZE],
            float bias[CODE_SIZE]) {
    L0:for(int row = 0; row < CODE_SIZE; row++) {
      float tmp = bias[row];  // Start with bias
      L1:for(int col = 0; col < HIDDEN_SIZE; col++) {
        tmp = tmp + weights[row][col] * layerInput[col];
      }
      // ReLu
      layerOutput[row] = max(0.0f, tmp);
    }
  }

void startDecryptionLayer(float layerInput[CODE_SIZE],
            float layerOutput[HIDDEN_SIZE],
            float weights[HIDDEN_SIZE][CODE_SIZE],
            float bias[HIDDEN_SIZE]) {
    L0:for(int row = 0; row < HIDDEN_SIZE; row++) {
      float tmp = bias[row];  // Start with bias
      L1:for(int col = 0; col < CODE_SIZE; col++) {
        tmp = tmp + weights[row][col] * layerInput[col];
      }
      // ReLu
      layerOutput[row] = max(0.0f, tmp);
    }
  }

void outputLayer(float layerInput[HIDDEN_SIZE],
            float layerOutput[INPUT_SIZE],
            float weights[INPUT_SIZE][HIDDEN_SIZE],
            float bias[INPUT_SIZE]) {
    L0:for(int row = 0; row < INPUT_SIZE; row++) {
      float tmp = bias[row];  // Start with bias
      L1:for(int col = 0; col < HIDDEN_SIZE; col++) {
        tmp = tmp + weights[row][col] * layerInput[col];
      }
      // ReLu
      layerOutput[row] = tmp;
    }
  }


