import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGenerator(nn.Module):
    def __init__(self, n_attributes, latent_dim, img_shape, ngf=64):
        super(DCGenerator, self).__init__()

        # self.label_emb = nn.Embedding(n_attributes, n_attributes)
        self.img_shape = img_shape

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(in_channels=latent_dim + n_attributes, out_channels=ngf * 8,
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
        gen_input = torch.cat((labels, noise), -1).unsqueeze(-1).unsqueeze(-1)
        img = self.model(gen_input)
        return img


class DCDiscriminator(nn.Module):
    def __init__(self, n_attributes, img_shape, n_embed=32, ndf=64):
        super(DCDiscriminator, self).__init__()

        self.label_embedding = nn.Linear(n_attributes, n_embed, bias=False)

        self.first_layer = nn.Sequential(
            # input is (3) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf) x 32 x 32
        )

        self.model = nn.Sequential(
            # state size. (ndf + n_attribs) x 32 x 32
            nn.Conv2d(ndf + n_embed, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
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
