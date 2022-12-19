import unittest

import torch

from src.model.blocks import ConvBlock


class TestConvBlock(unittest.TestCase):
    def setUp(self) -> None:
        self.channels, self.width, self.height = (128, 64, 64)

        self.conv_block_down = ConvBlock(self.channels, self.channels,
                                         kernel_size=3)
        self.conv_block_up = ConvBlock(self.channels, self.channels,
                                       kernel_size=4,
                                       downsample=False)
        self.image = torch.randn((self.channels, self.width, self.height))

    def test_downsampling_output_shape(self):
        expected_width = expected_height = self.width//2
        expected_output_shape = (self.channels, expected_width, expected_height)

        output = self.conv_block_down(self.image)
        self.assertEqual(output.shape, expected_output_shape)

    def test_upsampling_output_shape(self):
        expected_width = expected_height = self.width * 2
        expected_output_shape = (self.channels, expected_width, expected_height)

        output = self.conv_block_up(self.image)
        self.assertEqual(output.shape, expected_output_shape)
