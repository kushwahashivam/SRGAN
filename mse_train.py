import os
import pickle
from tqdm import tqdm, trange
import numpy as np
import torch
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter

from config import lr_size, hr_size
from models import Generator, Discriminator

gen_dir = "model/mse/generator.pt"
disc_dir = "model/mse/discriminator.pt"
gen_optim_dir = "model/mse/gen_optim.pt"
disc_optim_dir = "model/mse/disc_optim.pt"
gs_dir = "summary/mse_gs.pkl"

gs = 1
batch_size = 16
writer = SummaryWriter(log_dir="summary/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = Generator().to(device)
disc = Discriminator().to(device)
gen_optim = torch.optim.Adam(gen.parameters(), lr=10e-4, betas=(.9, .999))
disc_optim = torch.optim.Adam(disc.parameters(), lr=10e-4, betas=(.9, .999))
mse_criterion = torch.nn.MSELoss()
bce_criterion = torch.nn.BCELoss()

retrain = True
if retrain:
    print("Resuming training...")
    if os.path.exists(gs_dir):
        gs = pickle.load(open(gs_dir, "rb"))
    else:
        raise FileNotFoundError("Global step not found.")
    if os.path.exists(gen_dir):
        state_dict = torch.load(gen_dir)
        gen.load_state_dict(state_dict)
    else:
        raise FileNotFoundError("Generator model not found.")
    if os.path.exists(disc_dir):
        state_dict = torch.load(disc_dir)
        disc.load_state_dict(state_dict)
    else:
        raise FileNotFoundError("Discriminator model not found.")
    if os.path.exists(gen_optim_dir):
        state_dict = torch.load(gen_optim_dir)
        gen_optim.load_state_dict(state_dict)
    else:
        raise FileNotFoundError("Generator optimizer not found.")
    if os.path.exists(disc_optim_dir):
        state_dict = torch.load(disc_optim_dir)
        disc_optim.load_state_dict(state_dict)
    else:
        raise FileNotFoundError("Discriminator optimizer not found.")
    del state_dict

def train_disc(lr_img, labels):
    disc_optim.zero_grad()
    sr_img = gen(lr_img)
    pred = disc(sr_img)
    loss = bce_criterion(pred, labels)
    loss.backward()
    disc_optim.step()
    return loss.item()

def train_gen(lr_img, hr_img, labels):
    gen_optim.zero_grad()
    disc_optim.zero_grad()
    sr_img = gen(lr_img)
    pred = disc(sr_img)
    adv_loss = bce_criterion(pred, labels)
    mse_loss = mse_criterion(sr_img, hr_img)
    loss = 6e-3 * mse_loss + 10e-3 * adv_loss
    loss.backward()
    gen_optim.step()
    return loss.item()

num_epochs = 100000
for epoch in range(num_epochs):
    pass