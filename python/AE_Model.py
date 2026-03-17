# -*- coding: utf-8 -*-
"""
AE (AutoEncoder) model for MNIST database

Model adapted from https://github.com/nathanhubens/Autoencoders/blob/master/Autoencoders.ipynb
"""

import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, code_size):

        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, code_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(code_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):

        original_size = x.shape

        x = x.reshape(x.shape[0], -1)
        code = self.encoder(x)
        decoded = self.decoder(code)
        decoded = decoded.reshape(original_size)

        return decoded