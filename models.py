import torch
import torch.nn as nn
import torchvision as tv

from config import lr_size, hr_size

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
        self.conv[1].momentum = .8
        self.conv[4].momentum = .8
    
    def forward(self, x):
        z = self.conv(x)
        x = x + z
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4), 
            nn.PReLU(num_parameters=1, init=.25)
        )
        self.residuals = nn.ModuleList([ResidualBlock(64) for _ in range(16)])
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(64)
        )
        self.conv2[1].momentum = .8
        self.pixel_shuffler1 = nn.Sequential(
            nn.Conv2d(64, 64*self.r**2, kernel_size=3, stride=1, padding=1), 
            nn.PixelShuffle(upscale_factor=2), 
            nn.PReLU(num_parameters=1, init=.25)
        )
        self.pixel_shuffler2 = nn.Sequential(
            nn.Conv2d(64, 64*self.r**2, kernel_size=3, stride=1, padding=1), 
            nn.PixelShuffle(upscale_factor=2), 
            nn.PReLU(num_parameters=1, init=.25)
        )
        self.out = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4), 
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        z = None
        for residual in self.residuals:
            if z is None:
                z = residual(x)
            else:
                z = residual(z)
        z = self.conv2(z)
        x = x + z
        x = self.pixel_shuffler2(x)
        x = self.pixel_shuffler1(x)
        x = self.out(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(.2), 
        )
        self.convolutions = nn.ModuleList()
        conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(.2)
        )
        self.convolutions.append(conv)
        conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(.2)
        )
        self.convolutions.append(conv)
        conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(.2)
        )
        self.convolutions.append(conv)
        conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(.2)
        )
        self.convolutions.append(conv)
        conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(.2)
        )
        self.convolutions.append(conv)
        conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(.2)
        )
        self.convolutions.append(conv)
        conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(.2)
        )
        self.convolutions.append(conv)
        self.dense = nn.Sequential(
            nn.Linear((hr_size*hr_size*512)//(16*16), 1024), 
            nn.LeakyReLU(.2)
        )
        self.out = nn.Sequential(
            nn.Linear(1024, 1), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        for conv in self.convolutions:
            x = conv(x)
        x = x.view(x.size(0), (hr_size*hr_size*512)//(16*16))
        x = self.dense(x)
        x = self.out(x)
        return x


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tv.models.vgg19(pretrained=True)
        self.base = vgg.features[:35]
    
    def forward(self, x):
        return self.base(x)
