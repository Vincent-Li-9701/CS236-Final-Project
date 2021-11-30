import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter


from dad.model import *
from dad.data import *
from dad.config import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between image sampling")
parser.add_argument("--ckpt_interval", type=int, default=5, help="interval between checkpoint saving")
parser.add_argument("--save_name", type=str, default='milestone', help="name used to save this experie=ments")
opt = parser.parse_args()
print(opt)

os.makedirs("log", exist_ok=True)
os.makedirs(f"log/{opt.save_name}", exist_ok=True)
os.makedirs(f"log/{opt.save_name}/ckpt", exist_ok=True)
os.makedirs(f"log/{opt.save_name}/images", exist_ok=True)

img_shape = (3, *IMAGE_SIZE)

cuda = True if torch.cuda.is_available() else False

# Loss functions
adversarial_loss = nn.BCELoss()

# Initialize generator and discriminator
# cGAN
# generator = Generator(NUM_ATTRIBUTE, opt.latent_dim, img_shape)
# discriminator = Discriminator(NUM_ATTRIBUTE, img_shape)
# cDCGAN
generator = DCGenerator(NUM_ATTRIBUTE, opt.latent_dim, img_shape)
discriminator = DCDiscriminator(NUM_ATTRIBUTE, img_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
train_dataloader = torch.utils.data.DataLoader(
    data.DrivingImageDataset(folder_path=IMAGES_PATH, split='train', label_path=LABEL_PATH),
    batch_size=opt.batch_size,
    shuffle=True,
)
# val_dataloader = torch.utils.data.DataLoader(
#     data.DrivingImageDataset(folder_path=IMAGES_PATH, split='val', label_path=LABEL_PATH),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# Logging
writer = SummaryWriter(f'log/{opt.save_name}/run')


def sample_image(n_row, epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    labels = Variable(generate_random_attributes(n_row ** 2, FloatTensor, cuda))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, os.path.join("log", "images", f"{epoch}.png"), nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    if epoch != 0 and epoch % opt.sample_interval == 0:
        sample_image(n_row=10, epoch=epoch)

    if epoch != 0 and epoch % opt.ckpt_interval == 0:
        torch.save(generator.state_dict(), f'log/{opt.save_name}/ckpt/generator_{epoch}.pth')
        torch.save(discriminator.state_dict(),  f'log/{opt.save_name}/ckpt/discriminator_{epoch}.pth')

    for i, (imgs, labels) in enumerate(train_dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(FloatTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(generate_random_attributes(batch_size, FloatTensor, cuda))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Generate images with new generator
        # with torch.no_grad():
        #     gen_imgs_new = generator(z, gen_labels)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        D_x = validity_real.mean().item()
        D_G_z = validity_fake.mean().item()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f] [D(x): %.4f] [D(G(z)): %.4f]"
            % (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item(), g_loss.item(), D_x, D_G_z)
        )
        writer.add_scalar('D loss', d_loss.item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('G loss', g_loss.item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('D(x)', D_x, epoch * len(train_dataloader) + i)
        writer.add_scalar('D(G(z))', D_G_z, epoch * len(train_dataloader) + i)

