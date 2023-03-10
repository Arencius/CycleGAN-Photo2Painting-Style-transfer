import unittest

import torch

import config
import utils
from src.model.generator import Generator


class TestGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.channels, self.width, self.height = (3, 256, 256)

        self.generator = Generator().to(config.DEVICE)
        self.encoder = torch.nn.Sequential(self.generator.encoder_block1,
                                           self.generator.encoder_block2,
                                           self.generator.encoder_block3)
        self.bottleneck = self.generator.bottleneck
        self.decoder = torch.nn.Sequential(self.generator.decoder_block1,
                                           self.generator.decoder_block2,
                                           self.generator.decoder_block3)

        self.image = torch.randn((self.channels, self.width, self.height)).to(config.DEVICE)

    def test_encoder_output_shape(self):
        expected_channels = self.generator.filters * 4
        expected_width = expected_height = self.width // 4
        expected_output_shape = (expected_channels, expected_width, expected_height)

        encoder_output = self.encoder(self.image)

        self.assertEqual(encoder_output.shape, expected_output_shape)

    def test_bottleneck_output_shape(self):
        encoder_output = self.encoder(self.image)
        bottleneck_output = self.bottleneck(encoder_output)

        self.assertEqual(encoder_output.shape, bottleneck_output.shape)

    def test_generator_output_shape(self):
        generator_output = self.generator(self.image)

        self.assertEqual(generator_output.shape, self.image.shape)

    def test_generator_output_range(self):
        generator_output = self.generator(self.image)
        self.assertTrue(utils.is_tensor_in_range(generator_output))
