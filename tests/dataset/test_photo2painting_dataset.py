import os
import unittest
import random
import torch
import config
import utils

from src.dataset.photo2painting_dataset import Photo2PaintingDataset


class TestPhoto2PaintingDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Photo2PaintingDataset()
        self.random_index = random.randint(0, len(self.dataset))
        self.photo, self.painting = self.dataset[self.random_index]

    def test_dataset_length(self):
        expected_length = len(os.listdir(config.PAINTINGS_IMAGES_DIR))
        self.assertEqual(len(self.dataset), expected_length)

    def test_if_photos_and_paintings_images_have_equal_shape(self):
        self.assertEqual(self.photo.shape, self.painting.shape)

    def test_dataset_images_shape(self):
        expected_shape = (config.IMAGE_CHANNELS, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
        self.assertEqual(self.photo.shape, expected_shape)

    def test_dataset_images_value_range(self):
        self.assertTrue(utils.is_tensor_in_range(self.photo))
