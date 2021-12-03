import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class DCDiscriminator(nn.Module):
    def __init__(self, n_attributes, n_embed=32, ndf=32):
        super(DCDiscriminator, self).__init__()

        dropout_rate = 0.5

        self.first_layer = nn.Sequential(
            # input is (3) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf) x 32 x 32
        )

        self.model = nn.Sequential(
            # state size. (ndf + n_embed) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.adv_layer = nn.Sequential(nn.Linear(ndf * 8 * 4 ** 2, 1), nn.Sigmoid())

    def forward(self, img, labels):
        x = self.first_layer(img)
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)
        return validity

    def get_first_layer_gradnorm(self):
        return self.first_layer[0].weight.grad.detach().data.norm(2).item()

    def get_last_layer_gradnorm(self):
        return self.adv_layer[-2].weight.grad.detach().data.norm(2).item()


class DCGenerator(nn.Module):
    def __init__(self, n_attributes, latent_dim, n_embed=32, ngf=64):
        super(DCGenerator, self).__init__()

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=ngf * 8,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (3) x 64 x 64
            nn.ConvTranspose2d(in_channels=ngf, out_channels=3,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        img = self.model(noise.unsqueeze(-1).unsqueeze(-1))
        return img

    def get_first_layer_gradnorm(self):
        return self.model[0].weight.grad.detach().data.norm(2).item()

    def get_last_layer_gradnorm(self):
        return self.model[-2].weight.grad.detach().data.norm(2).item()
