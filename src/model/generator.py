import torch
import torch.nn as nn

from src.model.blocks import ConvBlock, ResidualBlock


class Generator(nn.Module):
    def __init__(self, res_blocks=6):
        """
        Generator
        :param res_blocks: number of residual blocks in the generator.
        """
        super().__init__()

        self.channels = 64
        self.encoder = nn.Sequential(
            ConvBlock(3, self.channels,
                      kernel_size=7,
                      padding=3),
            ConvBlock(self.channels, self.channels * 2,
                      kernel_size=3),
            ConvBlock(self.channels * 2, self.channels * 4,
                      kernel_size=3,
                      stride=1)
        )

        self.bottleneck = nn.Sequential(*[
            ResidualBlock(self.channels * 4, use_act=i > res_blocks // 2) for i in range(res_blocks)
        ])

        self.decoder = nn.Sequential(
            ConvBlock(self.channels * 4, self.channels * 4,
                      kernel_size=3,
                      stride=1,
                      downsample=False),
            ConvBlock(self.channels * 4, self.channels * 2,
                      kernel_size=4,
                      downsample=False),
            ConvBlock(self.channels * 2, self.channels,
                      kernel_size=4,
                      downsample=False),
            nn.Conv2d(self.channels, 3,
                      kernel_size=3,
                      padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)

        return torch.tanh(self.decoder(x))
