import os
from tqdm import tqdm, trange
import numpy as np
import torch
import torchvision as tv

from config import lr_size, hr_size
from models import Generator, Discriminator

gen_dir = "model/mse/generator.pt"
disc_dir = "model/mse/discriminator.pt"
gen_optim_dir = "model/mse/gen_optim.pt"
disc_optim_dir = "model/mse/disc_optim.pt"
gs = "summary/mse_gs.pkl"

