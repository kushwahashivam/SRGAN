import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(n_channels), 
            nn.PReLU(num_parameters=1, init=.25), 
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(n_channels)
        )
    
    def forward(self, x):
        z = self.conv(x)
        x = x + z
        return x