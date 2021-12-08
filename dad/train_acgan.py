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

import dad
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
parser.add_argument("--flip_label_interval", type=int, default=5, help="interval between label_fliping")
parser.add_argument("--skip_D_interval", type=int, default=1, help="interval between training Discriminator")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between image sampling")
parser.add_argument("--ckpt_interval", type=int, default=1, help="interval between checkpoint saving")
parser.add_argument("--save_name", type=str, default='milestone', help="name used to save this experie=ments")
opt = parser.parse_args()
print(opt)

# tricks to confuse the discriminator
FLIP_LABEL = True
ADD_NOISE = True

os.makedirs(LOG_PATH, exist_ok=True)
this_run_path = os.path.join(LOG_PATH, opt.save_name)
os.makedirs(this_run_path, exist_ok=True)
os.makedirs(os.path.join(this_run_path, "ckpt"), exist_ok=True)
os.makedirs(os.path.join(this_run_path, "images"), exist_ok=True)

img_shape = (3, *IMAGE_SIZE)

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'


# Configure data loader
train_dataset = data.DrivingImageDataset(folder_path=IMAGES_PATH, split='train', label_path=LABEL_PATH, use_key_attribs=True)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)
num_attributes = train_dataset.get_num_attribs()

# Loss functions
adversarial_loss = nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
# ACGAN
import dad.model.acgan
generator = dad.model.acgan.ACGenerator(num_attributes, opt.latent_dim)
generator.apply(weights_init)
discriminator = dad.model.acgan.ACDiscriminator(num_attributes)
discriminator.apply(weights_init)


if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# Logging
writer = SummaryWriter(f'{this_run_path}/run')


def sample_image(n_row, epoch, iteration):
    """Saves a grid of generated images varying weather, scene and time of day"""
    z = Variable(torch.randn(n_row * train_dataset.get_num_attribs(), opt.latent_dim, device=device))
    labels = Variable(train_dataset.generate_attributes_for_plotting(n_row, FloatTensor, cuda))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, f"{this_run_path}/images/epoch{epoch}_iter{iteration}.png", nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(train_dataloader):
        if epoch % opt.sample_interval == 0 and (i + 1) % 100 == 0:
            sample_image(n_row=8, epoch=epoch, iteration=i+1)

        batch_size = imgs.shape[0]

        # Adversarial ground truths with label smoothing
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        # Use Soft labels
        soft_valid = Variable(torch.rand(batch_size, 1, device=device) * 0.3 + 0.7, requires_grad=False)
        soft_fake = Variable(torch.rand(batch_size, 1, device=device) * 0.3, requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(FloatTensor))
        real_weather, real_scene, real_time = train_dataset.separate_label(labels)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(torch.randn(batch_size, opt.latent_dim, device=device))
        gen_labels = Variable(train_dataset.generate_random_attributes(batch_size, FloatTensor, cuda))
        gen_weather, gen_scene, gen_time = train_dataset.separate_label(gen_labels)

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_weather, pred_scene, pred_time = discriminator(gen_imgs)
        aux_loss = (auxiliary_loss(pred_weather, gen_weather) + auxiliary_loss(pred_scene, gen_scene) + auxiliary_loss(pred_time, gen_time)) / 3.
        g_loss = 0.5 * (adversarial_loss(validity, valid) + aux_loss)

        g_loss.backward()
        G_grad_first = generator.get_first_layer_gradnorm()
        G_grad_last = generator.get_last_layer_gradnorm()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        if i % opt.skip_D_interval != 0:
            continue

        # Add Gaussian noise
        gen_imgs_detached = gen_imgs.detach()
        if ADD_NOISE:
            disminishing = max((10. - epoch) / 10., 0.)
            noise = torch.randn(*real_imgs.size(), device=device) * 0.1 * disminishing
            real_imgs += noise
            gen_imgs_detached += noise

        if FLIP_LABEL and i % opt.flip_label_interval == 0:    # Label fliping
            soft_fake = Variable(torch.rand(batch_size, 1, device=device) * 0.3 + 0.7, requires_grad=False)
            soft_valid = Variable(torch.rand(batch_size, 1, device=device) * 0.3, requires_grad=False)

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, rpred_weather, rpred_scene, rpred_time = discriminator(real_imgs)
        real_aux_loss = (auxiliary_loss(rpred_weather, real_weather) + auxiliary_loss(rpred_scene, real_scene) + auxiliary_loss(rpred_time, real_time)) / 3.
        d_real_loss = (adversarial_loss(real_pred, soft_valid) + real_aux_loss) / 2.

        # Loss for fake images
        fake_pred, fpred_weather, fpred_scene, fpred_time = discriminator(gen_imgs_detached)
        fake_aux_loss = (auxiliary_loss(fpred_weather, gen_weather) + auxiliary_loss(fpred_scene, gen_scene) + auxiliary_loss(fpred_time, gen_time)) / 3.
        d_fake_loss = (adversarial_loss(fake_pred, soft_fake) + fake_aux_loss) / 2.

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2.

        d_loss.backward()
        D_grad_first = discriminator.get_first_layer_gradnorm()
        D_grad_last = discriminator.get_last_layer_gradnorm()
        optimizer_D.step()

        # Calculate discriminator accuracy
        all_real_pred = [x.data.cpu().numpy() for x in [rpred_weather, rpred_scene, rpred_time]]
        all_real_pred = np.concatenate([np.argmax(pred, axis=1) for pred in all_real_pred], axis=0)
        real_gt = np.concatenate([gt.data.cpu().numpy() for gt in [real_weather, real_scene, real_time]], axis=0)
        real_acc = np.mean(all_real_pred == real_gt)
        all_fake_pred = [x.data.cpu().numpy() for x in [fpred_weather, fpred_scene, fpred_time]]
        all_fake_pred = np.concatenate([np.argmax(pred, axis=1) for pred in all_fake_pred], axis=0)
        fake_gt = np.concatenate([gt.data.cpu().numpy() for gt in [gen_weather, gen_scene, gen_time]], axis=0)
        fake_acc = np.mean(all_fake_pred == fake_gt)

        D_x = real_pred.mean().item()
        D_G_z = fake_pred.mean().item()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f] [D acc: %.4f | %.4f] [D(x): %.4f] [D(G(z)): %.4f] [G_grad: %.4f / %.4f] [D_grad: %.4f / %.4f]"
            % (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item(), g_loss.item(), real_acc, fake_acc, D_x, D_G_z, G_grad_first, G_grad_last, D_grad_first, D_grad_last)
        )
        writer.add_scalar('D loss', d_loss.item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('G loss', g_loss.item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('D real acc', real_acc, epoch * len(train_dataloader) + i)
        writer.add_scalar('D fake acc', fake_acc, epoch * len(train_dataloader) + i)
        writer.add_scalar('D(x)', D_x, epoch * len(train_dataloader) + i)
        writer.add_scalar('D(G(z))', D_G_z, epoch * len(train_dataloader) + i)
        writer.add_scalar('G grad first', G_grad_first, epoch * len(train_dataloader) + i)
        writer.add_scalar('G grad last', G_grad_last, epoch * len(train_dataloader) + i)
        writer.add_scalar('D grad first', D_grad_first, epoch * len(train_dataloader) + i)
        writer.add_scalar('D grad last', D_grad_last, epoch  * len(train_dataloader) + i)

    if epoch % opt.ckpt_interval == 0:
        torch.save(generator.state_dict(), f'{this_run_path}/ckpt/generator_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'{this_run_path}/ckpt/discriminator_{epoch}.pth')
