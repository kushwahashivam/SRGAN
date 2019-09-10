import os
from PIL import Image
import numpy as np
import torch
import torchvision as tv

from config import hr_size, lr_size

class Div2KDataset(torch.utils.data.Dataset):
    def __init__(self, root, num_samples):
        self.root = root
        self.num_samples = num_samples
        self.imgs = [os.path.join(self.root, img) for img in os.listdir(self.root)]
        self.indices = np.random.randint(low=0, high=len(self.imgs), size=self.num_samples)
        self.hr_transform = tv.transforms.RandomCrop(hr_size)
        self.lr_transform = tv.transforms.Resize(lr_size)
        self.img_transform = tv.transforms.ToTensor()
        self.label_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(), 
            tv.transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
        ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        img = Image.open(self.imgs[self.indices[index]])
        hr_img = self.hr_transform(img)
        lr_img = self.lr_transform(hr_img)
        hr_img = self.label_transform(hr_img)
        lr_img = self.img_transform(lr_img)
        return lr_img, hr_img
