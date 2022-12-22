import unittest

import torch

from src.model.blocks import ConvBlock


class TestConvBlock(unittest.TestCase):
    def setUp(self) -> None:
        self.channels, self.width, self.height = (128, 64, 64)

        self.conv_block_down_relu = ConvBlock(self.channels, self.channels,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1)
        self.conv_block_down_leaky_relu = ConvBlock(self.channels, self.channels,
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    leaky_relu=True)

        self.conv_block_up = ConvBlock(self.channels, self.channels,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1,
                                       upsample=True)
        self.image = torch.randn((self.channels, self.width, self.height))

    def test_downsampling_output_shape(self):
        expected_width = expected_height = self.width // 2
        expected_output_shape = (self.channels, expected_width, expected_height)

        output = self.conv_block_down_relu(self.image)
        self.assertEqual(expected_output_shape, output.shape)

    def test_upsampling_output_shape(self):
        expected_width = expected_height = self.width * 2
        expected_output_shape = (self.channels, expected_width, expected_height)

        output = self.conv_block_up(self.image)
        self.assertEqual(expected_output_shape, output.shape)

    def test_conv_block_relu_activation(self):
        output = self.conv_block_down_relu(self.image)
        self.assertTrue(torch.all(output >= 0))

    def test_conv_block_leaky_relu_activation(self):
        output = self.conv_block_down_leaky_relu(self.image)
        self.assertFalse(torch.all(output >= 0))