import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class ACDiscriminator(nn.Module):
    def __init__(self, n_attributes, n_embed=32, ndf=32, weather_classes=4, scene_classes=3, time_classes=3):
        super(ACDiscriminator, self).__init__()

        dropout_rate = 0.5

        self.model = nn.Sequential(
            # input is (3) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
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
            # nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
            # state size. (ndf*8) x 1 x 1
        )
        self.adv_layer = nn.Sequential(nn.Linear(128 * 4 ** 2, 1), nn.Sigmoid())
        self.weather_layer = nn.Sequential(nn.Linear(128 * 4 ** 2, weather_classes), nn.Softmax())
        self.scene_layer = nn.Sequential(nn.Linear(128 * 4 ** 2, scene_classes), nn.Softmax())
        self.time_layer = nn.Sequential(nn.Linear(128 * 4 ** 2, time_classes), nn.Softmax())


    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        weather = self.weather_layer(out)
        scene = self.scene_layer(out)
        time = self.time_layer(out)
        return validity, weather, scene, time

    def get_first_layer_gradnorm(self):
        return self.model[0].weight.grad.detach().data.norm(2).item()

    def get_last_layer_gradnorm(self):
        return self.model[-4].weight.grad.detach().data.norm(2).item()


class ACGenerator(nn.Module):
    def __init__(self, n_attributes, latent_dim, n_embed=32, ngf=64):
        super(ACGenerator, self).__init__()

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


class ACGeneratorAlt(nn.Module):
    def __init__(self, n_attributes, latent_dim, n_embed=32, ngf=64):
        super(ACGenerator, self).__init__()

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
