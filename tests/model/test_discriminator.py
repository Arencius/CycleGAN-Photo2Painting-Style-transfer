import unittest

import torch
import config

from src.model.discriminator import Discriminator


class TestDiscriminator(unittest.TestCase):
    def setUp(self) -> None:
        self.discriminator = Discriminator()
        self.image = torch.randn((3, 256, 256))

    def test_discriminator_output_shape(self):
        output = self.discriminator(self.image)

        expected_output_shape = (1, config.DISCRIMINATOR_PATCH_SIZE, config.DISCRIMINATOR_PATCH_SIZE)
        self.assertEqual(expected_output_shape, output.shape)

