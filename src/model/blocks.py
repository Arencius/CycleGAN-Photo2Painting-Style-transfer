import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self,
                 channels: int = 256,
                 use_act: bool = False):
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


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride=2,
                 padding=1,
                 downsample=True):
        """
        Convolutional block used in encoder and decoder part of the generator model.
        :param in_channels: number of channels in the input image
        :param out_channels: number of output channels
        :param kernel_size: size of the kernel
        :param stride: value of stride (in this case it's 2 for downsampling and upsampling the image)
        :param padding: value of padding applied to the image
        :param downsample: indicates whether the image gets down or upsampled
        """
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding) if downsample
            else nn.ConvTranspose2d(in_channels, out_channels,
                                    kernel_size,
                                    stride=stride,
                                    padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)
