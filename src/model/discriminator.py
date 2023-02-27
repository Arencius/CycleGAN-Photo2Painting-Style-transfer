import torch
import torch.nn as nn

import config
from src.model.blocks import ConvBlock


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.filters = 64

        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.filters,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(config.ALPHA)
        )

        self.model = nn.Sequential(
            ConvBlock(self.filters, self.filters * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      leaky_relu=True),
            ConvBlock(self.filters * 2, self.filters * 4,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      leaky_relu=True),
            ConvBlock(self.filters * 4, self.filters * 8,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      leaky_relu=True)
        )
        self.output = nn.Conv2d(self.filters * 8, 1,
                                kernel_size=4,
                                stride=1,
                                padding=3,
                                padding_mode="reflect"
                                )

    def forward(self, x):
        x = self.initial_block(x)
        x = self.model(x)
        return torch.sigmoid(self.output(x))
