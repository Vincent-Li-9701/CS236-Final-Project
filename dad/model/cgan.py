import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Generator(nn.Module):
    def __init__(self, n_attributes, latent_dim, img_shape, n_embed=32):
        super(Generator, self).__init__()

        self.label_embedding = nn.Linear(n_attributes, n_embed, bias=False)
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_embed, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        embedding = self.label_embedding(labels)
        gen_input = torch.cat((embedding, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

    def get_first_layer_gradnorm(self):
        return self.model[0].weight.grad.detach().data.norm(2).item()

    def get_last_layer_gradnorm(self):
        return self.model[-2].weight.grad.detach().data.norm(2).item()


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
