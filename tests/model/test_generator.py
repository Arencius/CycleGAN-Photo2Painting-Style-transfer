import unittest

import torch

import config
from src.model.generator import Generator


class TestGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.channels, self.width, self.height = (3, 256, 256)

        self.generator = Generator().to(config.DEVICE)
        #self.encoder = self.generator.encoder
        #self.bottleneck = self.generator.bottleneck
        #self.decoder = self.generator.decoder

        self.image = torch.randn((self.channels, self.width, self.height))

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

    def test_generator_summary(self):
        from torchsummary import summary

        summary(self.generator, (3,256,256))

    def test_generator_output_range(self):
        generator_output = self.generator(self.image)

        is_output_greater_than_negative_one = torch.all(generator_output >= -1.0)
        is_output_smaller_than_one = torch.all(generator_output <= 1)
        is_output_in_range = torch.stack([is_output_greater_than_negative_one,
                                          is_output_smaller_than_one])

        self.assertTrue(torch.all(is_output_in_range))
