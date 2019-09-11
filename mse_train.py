import os
import pickle
from tqdm import tqdm, trange
import numpy as np
import torch
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter

from config import lr_size, hr_size
from utils import Div2KDataset
from models import Generator, Discriminator

root_train = "data/DIV2K_train_HR/"
root_val = "data/DIV2K_valid_HR/"
gen_dir = "model/mse/generator.pt"
disc_dir = "model/mse/discriminator.pt"
gen_optim_dir = "model/mse/gen_optim.pt"
disc_optim_dir = "model/mse/disc_optim.pt"
gs_dir = "summary/mse_gs.pkl"
log_dir = "summary/"

gs = 1
batch_size = 16
writer = SummaryWriter(log_dir=log_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = Generator().to(device)
disc = Discriminator().to(device)
gen_optim = torch.optim.Adam(gen.parameters(), lr=10e-4, betas=(.9, .999))
disc_optim = torch.optim.Adam(disc.parameters(), lr=10e-4, betas=(.9, .999))
mse_criterion = torch.nn.MSELoss()
bce_criterion = torch.nn.BCELoss()

retrain = False
if retrain:
    print("Resuming training...")
    if os.path.exists(gs_dir):
        gs = pickle.load(open(gs_dir, "rb"))
    else:
        raise FileNotFoundError("Global step not found.")
    if os.path.exists(gen_dir):
        state_dict = torch.load(gen_dir, map_location=device)
        gen.load_state_dict(state_dict)
    else:
        raise FileNotFoundError("Generator model not found.")
    if os.path.exists(disc_dir):
        state_dict = torch.load(disc_dir, map_location=device)
        disc.load_state_dict(state_dict)
    else:
        raise FileNotFoundError("Discriminator model not found.")
    if os.path.exists(gen_optim_dir):
        state_dict = torch.load(gen_optim_dir, map_location=device)
        gen_optim.load_state_dict(state_dict)
    else:
        raise FileNotFoundError("Generator optimizer not found.")
    if os.path.exists(disc_optim_dir):
        state_dict = torch.load(disc_optim_dir, map_location=device)
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
    loss = mse_loss + 10e-3 * adv_loss
    loss.backward()
    gen_optim.step()
    return loss.item()

print("Generating dataset...")
num_epochs = 100000
ds_train = Div2KDataset(root_train, num_epochs)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
ds_val = Div2KDataset(root_val, num_epochs)
dl_val = torch.utils.data.DataLoader(ds_val, batch_size=4, shuffle=True, drop_last=True, num_workers=0)

print("Training started.")
for lr_img, hr_img in tqdm(dl_train):
    lr_img = lr_img.to(device)
    hr_img = hr_img.to(device)
    # Train discriminator on real data
    disc_optim.zero_grad()
    labels = torch.ones(batch_size, 1).to(device)
    pred = disc(hr_img)
    loss = bce_criterion(pred, labels)
    loss.backward()
    disc_optim.step()
    loss = loss.item()
    writer.add_scalar("mse_training/loss/disc_loss_real", loss, gs)
    # Train discriminator on fake data
    labels = torch.zeros(batch_size, 1).to(device)
    loss = train_disc(lr_img, labels)
    writer.add_scalar("mse_training/loss/disc_loss_fake", loss, gs)
    # Train generator
    labels = torch.ones(batch_size, 1).to(device)
    loss = train_gen(lr_img, hr_img, labels)
    writer.add_scalar("mse_training/loss/gen_loss", loss, gs)
    if (gs)%100 == 0:
        # Save state of training
        pickle.dump(gs, open(gs_dir, "wb"))
        torch.save(gen.state_dict(), gen_dir)
        torch.save(disc.state_dict(), disc_dir)
        torch.save(gen_optim.state_dict(), gen_optim_dir)
        torch.save(disc_optim.state_dict(), disc_optim_dir)
        # Visualize some generated images
        gen.eval()
        lr_img, hr_img = next(iter(dl_val))
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        sr_img = gen(lr_img)
        # Scale the images to [0, 1]
        hr_img = hr_img * .5 + .5
        sr_img = sr_img * .5 + .5
        writer.add_image("mse_training/image/original_images", tv.utils.make_grid(hr_img, nrow=2), 0)
        writer.add_image("mse_training/image/superresolved_images", tv.utils.make_grid(sr_img, nrow=2), 0)
        gen.train()
        writer.flush()
    gs += 1
writer.flush()
writer.close()