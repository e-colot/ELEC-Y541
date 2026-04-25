// AE_test.cpp
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "BMPProcessor.h"
#include "AE.h"
#include "constant.h"

// I/O Image Path
#define INPUT_IMAGE_BASE "Test_Images/digit7.bmp"
#define OUTPUT_IMAGE_BASE "outputAE_7.bmp"

int in_pix[MAX_HEIGHT][MAX_WIDTH];
int out_pix[MAX_HEIGHT][MAX_WIDTH];

int main() {
    std::cout << "=======================================================" << std::endl;
    std::cout << "--------- Simulation started ---------" << std::endl<< std::endl;

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

    int out[MAX_HEIGHT][MAX_WIDTH];
    
    // Apply auto-encoder
    AE(in_pix, out_pix, enc1, enc1_bias, enc2, enc2_bias, enc3, enc3_bias, enc4, enc4_bias, dec1, dec1_bias, dec2, dec2_bias, dec3, dec3_bias, dec4, dec4_bias);

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
