import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class cHCGenerator(nn.Module):
    def __init__(self, n_attributes, latent_dim, img_size, n_embed=32, ngf=64):
        super(cHCGenerator, self).__init__()

        self.label_embedding = nn.Linear(n_attributes, n_embed, bias=True)

        self.init_size = 16
        self.ngf = ngf
        self.l1 = nn.Linear(latent_dim + n_embed, ngf * 2 * self.init_size ** 2)

        self.model = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 2, 5, stride=1, padding=2),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 5, stride=1, padding=2),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, 3, 5, stride=1, padding=2),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        embedding = self.label_embedding(labels)
        inputs = torch.cat((noise, embedding), -1)
        out = self.l1(inputs)
        out = out.view(out.shape[0], self.ngf * 2, self.init_size, self.init_size)
        img = self.model(out)
        return img

    def get_first_layer_gradnorm(self):
        return self.l1.weight.grad.detach().data.norm(2).item()

    def get_last_layer_gradnorm(self):
        return self.model[-2].weight.grad.detach().data.norm(2).item()


class cDCDiscriminator(nn.Module):
    def __init__(self, n_attributes, n_embed=32, ndf=32):
        super(cDCDiscriminator, self).__init__()

        self.label_embedding = nn.Linear(n_attributes, n_embed, bias=True)

        dropout_rate = 0.5

        self.first_layer = nn.Sequential(
            # input is (3) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf) x 32 x 32
        )

        self.model = nn.Sequential(
            # state size. (ndf + n_attribs) x 32 x 32
            nn.Conv2d(ndf + n_embed, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. (ndf*8) x 1 x 1
        )

    def forward(self, img, labels):
        x = self.first_layer(img)
        embedding = self.label_embedding(labels)
        # tile labels so that it has the same shape as the output of first conv layer
        tiled_embeds = torch.tile(embedding.unsqueeze(-1).unsqueeze(-1), (1, 1, 32, 32))
        x = torch.cat((x, tiled_embeds), dim=1)
        validity = self.model(x)[:, :, 0, 0]
        return validity

    def get_first_layer_gradnorm(self):
        return self.first_layer[0].weight.grad.detach().data.norm(2).item()

    def get_last_layer_gradnorm(self):
        return self.model[-2].weight.grad.detach().data.norm(2).item()
