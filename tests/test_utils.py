import os
import unittest

import config
import utils
import random


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.steps = 15
        self.epoch = 127
        self.dummy_losses = [[random.randint(0, 1560) for _ in range(self.steps)] for _ in range(2)]

    def test_if_saves_plot_in_correct_directory(self):
        path = os.path.join(config.LOSS_PLOTS_DIR, f'{self.epoch+1}.png')

        if os.path.exists(path):
            os.remove(path)

        utils.save_epoch_loss_results(self.epoch, self.dummy_losses)

        self.assertTrue(f'{self.epoch+1}.png' in os.listdir(config.LOSS_PLOTS_DIR))
