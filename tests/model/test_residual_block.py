import unittest

import torch

from src.model.blocks import ResidualBlock


class TestResidualBlock(unittest.TestCase):
    def setUp(self) -> None:
        self.residual_block = ResidualBlock()
        self.conv_layers = self.residual_block.block  # conv block used to check if skip connections work properly
        self.image = torch.randn((256, 64, 64))

    def test_forward_output_shape(self):
        output = self.residual_block(self.image)
        self.assertEqual(self.image.shape, output.shape)

    def test_if_skip_connection_works(self):
        conv_block_output = self.conv_layers(self.image)
        res_block_output = self.residual_block(self.image)

        diff = res_block_output - conv_block_output
        torch.testing.assert_close(diff, self.image)

    def test_residual_block_with_activation(self):
        residual_block_with_act = ResidualBlock(use_act=True)

        output = residual_block_with_act(self.image)
        self.assertTrue(torch.all(output >= 0))
