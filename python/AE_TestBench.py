# -*- coding: utf-8 -*-
"""
Test Bench for AutoEncoder
"""    
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import cv2
import numpy as np

from AE_Model import AutoEncoder


# ============= AE Model Testing ====================
if __name__ == '__main__':

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # ======== define Input image location ====================
    filename = 'Test_Images/digit3.bmp' # change digit number here (0~9)
    # =========================================================

    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)/255.0
    image = image[:, :, np.newaxis] # Adding new dimension for the color channel

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    # ======== Loading Model and its parameters ===============
    model = AutoEncoder(784, 256, 128).to(device)
    model.load_state_dict(torch.load('mnist_autoencoder.pt', map_location=device))
    # =========================================================
    
    # Testing 
    model.eval()
    with torch.no_grad():
        data = transform(image)
        data = data.unsqueeze(0)
        target = data.clone()
        data, target = data.to(device), data.to(device)

        output = model(data)
        mse = F.mse_loss(output, target).item()

        print('Reconstruction error (mse): {}'.format(mse))

        # Visualizing output
        cv2.imwrite('output.bmp', output[0, 0].detach().cpu().numpy()*255)

