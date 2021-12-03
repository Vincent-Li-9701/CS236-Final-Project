import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, n_attributes, img_shape, n_embed=32):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Linear(n_attributes, n_embed, bias=False)

        self.model = nn.Sequential(
            nn.Linear(n_embed + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        embedding = self.label_embedding(labels)
        d_in = torch.cat((img.view(img.size(0), -1), embedding), -1)
        validity = self.model(d_in)
        return validity

    def get_first_layer_gradnorm(self):
        return self.model[0].weight.grad.detach().data.norm(2).item()

    def get_last_layer_gradnorm(self):
        return self.model[-2].weight.grad.detach().data.norm(2).item()


class cDCGenerator(nn.Module):
    def __init__(self, n_attributes, latent_dim, n_embed=32, ngf=64):
        super(cDCGenerator, self).__init__()

        self.label_embedding = nn.Linear(n_attributes, n_embed, bias=False)

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(in_channels=latent_dim + n_embed, out_channels=ngf * 8,
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
        # Concatenate label embedding and image to produce input
        embedding = self.label_embedding(labels)
        gen_input = torch.cat((noise, embedding), -1).unsqueeze(-1).unsqueeze(-1)
        img = self.model(gen_input)
        return img

    def get_first_layer_gradnorm(self):
        return self.model[0].weight.grad.detach().data.norm(2).item()

    def get_last_layer_gradnorm(self):
        return self.model[-2].weight.grad.detach().data.norm(2).item()
