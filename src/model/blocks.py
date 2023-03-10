import torch
import torch.nn as nn

import config


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 upsample=False,
                 use_act=True,
                 leaky_relu=False,
                 reflection_padding=True,
                 **kwargs):
        """
        Convolutional block used in encoder and decoder part of the generator model.
        :param in_channels: number of channels in the input image
        :param out_channels: number of output channels
        :param kernel_size: size of the kernel
        :param upsample: indicates whether the image gets upsampled
        :param use_act: whether to use activation function in the block
        :param leaky_relu: activation function used in the block - if True, LeakyReLU is used, otherwise ReLU
        :param reflection_padding: whether to use reflection or zeros padding in the convolution
        """
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size,
                      padding_mode='reflect' if reflection_padding else 'zeros',
                      **kwargs) if not upsample
            else nn.ConvTranspose2d(in_channels, out_channels,
                                    kernel_size,
                                    **kwargs),
            nn.InstanceNorm2d(out_channels)
        )

        self.activation = nn.Identity() if not use_act \
            else nn.LeakyReLU(config.ALPHA) if leaky_relu else nn.ReLU()

    def forward(self, x, skip_block_output: torch.Tensor = None):
        """
        Forward pass of the block. Applies a convolutional block to the input tensor and adds a
        skip connection with the output of the corresponding encoder block (if provided), before applying
        an activation function.

        :param x: input tensor to be processed by the decoder block.
        :param skip_block_output: optional tensor representing the output of the corresponding encoder block.
        :return: output tensor of the decoder block after applying the convolutional block, skip connection, and
                 activation function.
        """
        conv_block_output = self.conv_block(x)

        if skip_block_output is not None:
            conv_block_output += skip_block_output

        return self.activation(conv_block_output)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels=256):
        """
        Convolutional block with skip connection.
        :param in_channels: number of channels in the input image
        :param out_channels: number of output channels
        """
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      reflection_padding=False),
            ConvBlock(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      use_act=False,
                      reflection_padding=False)
        )

    def forward(self, x):
        return x + self.block(x)
