import torch
import torch.nn as nn
import pandas as pd


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1),
                        nn.ELU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 2, padding = 1),
                        nn.ELU())
        self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(out_channels, out_channels, stride = 2, kernel_size = 4, padding = 1),
                        nn.ELU(),
                        nn.ConvTranspose2d(out_channels, in_channels, stride = 2, kernel_size = 4, padding = 1),
                        nn.ELU())
        
    def forward(self, x):
        for l in self.encoder:
            x = l(x)
        for j in self.decoder:
            x = j(x)
        return x
    
    def encode(self, x):
        for l in self.encoder:
            x = l(x)
        return x
    
    def decode(self, x):
        for j in self.decoder:
            x = j(x)
        return x
    
