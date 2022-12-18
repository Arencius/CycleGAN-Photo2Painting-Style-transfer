import torch
import torch.nn as nn
from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 256, use_act: bool = False):
        """
        Convolutional block with skip connection.
        :param channels: number of channels in the conv layers within the block.
        """
        super().__init__()
        self.use_act = use_act

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=3,
                      padding=1),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=3,
                      padding=1)
        )

    def forward(self, x):
        act = nn.ReLU() if self.use_act else nn.Identity()
        return act(x + self.block(x))
