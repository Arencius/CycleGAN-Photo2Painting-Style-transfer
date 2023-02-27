import os
import unittest

import torch
import config
import utils
import random


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.steps = 150
        self.epoch = 512
        self.dummy_losses = [[random.randint(0, 1560) for _ in range(self.steps)] for _ in range(2)]
        self.tensor = torch.rand(3, 256, 256) * 2

    def test_if_saves_plot_in_correct_directory(self):
        path = os.path.join(config.LOSS_PLOTS_DIR, f'{self.epoch+1}.png')

        if os.path.exists(path):
            os.remove(path)

        utils.save_epoch_loss_results(self.epoch, self.dummy_losses)

        self.assertTrue(f'{self.epoch+1}.png' in os.listdir(config.LOSS_PLOTS_DIR))

    def test_tensor_is_in_range_correct_input(self):
        t = self.tensor - 1
        self.assertTrue(utils.is_tensor_in_range(t))

    def test_tensor_is_in_range_wrong_output(self):
        self.assertFalse(utils.is_tensor_in_range(self.tensor))
