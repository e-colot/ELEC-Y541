// AE_test.cpp
#include <stdio.h>
#include <stdlib.h>
#include "BMPProcessor.h"
#include "AE.h"

// I/O Image Path
#define INPUT_IMAGE_BASE "digit0.bmp"
#define OUTPUT_IMAGE_BASE "outputAE.bmp"

int in_pix[MAX_HEIGHT][MAX_WIDTH];
int out_pix[MAX_HEIGHT][MAX_WIDTH];

void extractWeights(FILE* file, float weight[INPUT_SIZE*HIDDEN_SIZE], size_t expectedSize) {
    
    if (!file) {
        perror("Failed to open a weight file");
        return;
    }
    
    size_t readSize = fread(weight, sizeof(float), expectedSize, file);
    fclose(file);

    if (readSize != expectedSize) {
        fprintf(stderr, "Got %zu floats, expected %zu\n", readSize, expectedSize);
    }

}

void extractBias(FILE* file, float bias[HIDDEN_SIZE], size_t expectedSize) {
    
    if (!file) {
        perror("Failed to open a bias file");
        return;
    }
    
    size_t readSize = fread(bias, sizeof(float), expectedSize, file);
    fclose(file);

    if (readSize != expectedSize) {
        fprintf(stderr, "Got %zu floats, expected %zu\n", readSize, expectedSize);
    }

}

int main() {
    std::cout << "=======================================================" << std::endl;
    std::cout << "--------- Simulation satrted ---------" << std::endl<< std::endl;

    // Load input image
    image::BMPMini bmp(INPUT_IMAGE_BASE);
    auto img = bmp.get();
    unsigned char *Grey = (unsigned char *)malloc(MAX_HEIGHT* MAX_WIDTH * sizeof(unsigned char));
    bmp.getGrayChannel(Grey);

    for (int x = 0; x < MAX_HEIGHT; x++) {
        for (int y = 0; y < MAX_WIDTH; y++) {
            in_pix[x][y] = Grey[x * MAX_WIDTH + y];
        }
    }

    // Load weights
    float enc1[INPUT_SIZE*HIDDEN_SIZE]; 
    float enc2[INPUT_SIZE*HIDDEN_SIZE]; 
    float enc3[INPUT_SIZE*HIDDEN_SIZE]; 
    float enc4[INPUT_SIZE*HIDDEN_SIZE]; 
    float dec1[INPUT_SIZE*HIDDEN_SIZE]; 
    float dec2[INPUT_SIZE*HIDDEN_SIZE]; 
    float dec3[INPUT_SIZE*HIDDEN_SIZE]; 
    float dec4[INPUT_SIZE*HIDDEN_SIZE];
    
    // Load biases
    float enc1_bias[INPUT_SIZE];
    float enc2_bias[INPUT_SIZE];
    float enc3_bias[INPUT_SIZE];
    float enc4_bias[INPUT_SIZE];
    float dec1_bias[INPUT_SIZE];
    float dec2_bias[INPUT_SIZE];
    float dec3_bias[INPUT_SIZE];
    float dec4_bias[INPUT_SIZE];

    FILE* f1 = fopen("enc1.weight", "rb");
    extractWeights(f1, enc1, INPUT_SIZE*HIDDEN_SIZE);
    FILE* f2 = fopen("enc2.weight", "rb");
    extractWeights(f2, enc2, HIDDEN_SIZE*HIDDEN_SIZE);
    FILE* f3 = fopen("enc3.weight", "rb");
    extractWeights(f3, enc3, HIDDEN_SIZE*HIDDEN_SIZE);
    FILE* f4 = fopen("enc4.weight", "rb");
    extractWeights(f4, enc4, CODE_SIZE*HIDDEN_SIZE);

    FILE* f5 = fopen("dec1.weight", "rb");
    extractWeights(f5, dec1, CODE_SIZE*HIDDEN_SIZE);
    FILE* f6 = fopen("dec2.weight", "rb");
    extractWeights(f6, dec2, HIDDEN_SIZE*HIDDEN_SIZE);
    FILE* f7 = fopen("dec3.weight", "rb");
    extractWeights(f7, dec3, HIDDEN_SIZE*HIDDEN_SIZE);
    FILE* f8 = fopen("dec4.weight", "rb");
    extractWeights(f8, dec4, INPUT_SIZE*HIDDEN_SIZE);
    
    // Load biases
    FILE* fb1 = fopen("enc1.bias", "rb");
    extractBias(fb1, enc1_bias, HIDDEN_SIZE);
    FILE* fb2 = fopen("enc2.bias", "rb");
    extractBias(fb2, enc2_bias, HIDDEN_SIZE);
    FILE* fb3 = fopen("enc3.bias", "rb");
    extractBias(fb3, enc3_bias, HIDDEN_SIZE);
    FILE* fb4 = fopen("enc4.bias", "rb");
    extractBias(fb4, enc4_bias, CODE_SIZE);
    
    FILE* fb5 = fopen("dec1.bias", "rb");
    extractBias(fb5, dec1_bias, HIDDEN_SIZE);
    FILE* fb6 = fopen("dec2.bias", "rb");
    extractBias(fb6, dec2_bias, HIDDEN_SIZE);
    FILE* fb7 = fopen("dec3.bias", "rb");
    extractBias(fb7, dec3_bias, HIDDEN_SIZE);
    FILE* fb8 = fopen("dec4.bias", "rb");
    extractBias(fb8, dec4_bias, INPUT_SIZE);

    int out[MAX_HEIGHT][MAX_WIDTH];
    
    // Apply auto-encoder
    AE(in_pix, out_pix, 
       enc1, enc2, enc3, enc4,
       enc1_bias, enc2_bias, enc3_bias, enc4_bias,
       dec1, dec2, dec3, dec4,
       dec1_bias, dec2_bias, dec3_bias, dec4_bias);

    // Write back the output image to the file
    for (int x = 0; x < MAX_HEIGHT; x++) {
        for (int y = 0; y < MAX_WIDTH; y++) {
            Grey[x * MAX_WIDTH + y] = out_pix[x][y] & 0xff;
        }
    }

    bmp.writeGreyScale(OUTPUT_IMAGE_BASE, Grey, MAX_WIDTH, MAX_HEIGHT);
    std::cout << " 8-Bit GreyScale Image Written " << std::endl;

    // Free allocated memory
    free(Grey);

    std::cout << std::endl<< "--------- Simulation END ---------" << std::endl;
    std::cout << "=======================================================" << std::endl;
    return 0;
}
