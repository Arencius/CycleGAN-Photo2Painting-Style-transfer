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

        self.encoder_block1 = ConvBlock(3, self.filters,
                                        kernel_size=7,
                                        padding=3,
                                        stride=1)
        self.encoder_block2 = ConvBlock(self.filters, self.filters * 2,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)
        self.encoder_block3 = ConvBlock(self.filters * 2, self.filters * 4,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)

        self.bottleneck = nn.Sequential(*[
            ResidualBlock(in_channels=self.filters * 4) for _ in range(res_blocks)
        ])

        self.decoder_block1 = ConvBlock(self.filters * 4, self.filters * 4,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        upsample=True,
                                        reflection_padding=False)
        self.decoder_block2 = ConvBlock(self.filters * 4, self.filters * 2,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        upsample=True,
                                        reflection_padding=False)
        self.decoder_block3 = ConvBlock(self.filters * 2, self.filters,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        upsample=True,
                                        reflection_padding=False)
        self.output = nn.Conv2d(self.filters, 3,
                                kernel_size=7,
                                padding=3,
                                padding_mode='reflect')

    def forward(self, x):
        encoder_block_output1 = self.encoder_block1(x)
        encoder_block_output2 = self.encoder_block2(encoder_block_output1)
        encoder_block_output3 = self.encoder_block3(encoder_block_output2)

        bottleneck_output = self.bottleneck(encoder_block_output3)

        decoder_block_output1 = self.decoder_block1(bottleneck_output, encoder_block_output3)
        decoder_block_output2 = self.decoder_block2(decoder_block_output1, encoder_block_output2)
        decoder_block_output3 = self.decoder_block3(decoder_block_output2, encoder_block_output1)

        return torch.tanh(self.output(decoder_block_output3))
