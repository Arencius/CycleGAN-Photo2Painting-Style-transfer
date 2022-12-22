import torch
import torch.nn as nn

from src.model.blocks import ConvBlock, ResidualBlock


class Generator(nn.Module):
    def __init__(self, res_blocks=6):
        """
        CycleGAN generator model.
        :param res_blocks: number of residual blocks in the generator.
        """
        super().__init__()

        self.filters = 64
        self.encoder = nn.Sequential(
            ConvBlock(3, self.filters,
                      kernel_size=7,
                      padding=3,
                      stride=2),
            ConvBlock(self.filters, self.filters * 2,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            ConvBlock(self.filters * 2, self.filters * 4,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        )

        self.bottleneck = nn.Sequential(*[
            ResidualBlock(self.filters * 4) for _ in range(res_blocks)
        ])

        self.decoder = nn.Sequential(
            ConvBlock(self.filters * 4, self.filters * 4,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      upsample=True),
            ConvBlock(self.filters * 4, self.filters * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      upsample=True),
            ConvBlock(self.filters * 2, self.filters,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      upsample=True)
        )
        self.output = nn.Conv2d(self.filters, 3,
                                kernel_size=7,
                                padding=3,
                                padding_mode='reflect')

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return torch.tanh(self.output(x))
