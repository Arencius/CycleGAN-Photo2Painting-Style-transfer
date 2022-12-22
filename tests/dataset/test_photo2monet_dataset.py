import os
import unittest
import random
import torch

import config
from src.dataset.photo2monet_dataset import Photo2MonetDataset


class TestPhoto2MonetDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Photo2MonetDataset()
        self.random_index = random.randint(0, len(self.dataset))
        self.photo, self.monet = self.dataset[self.random_index]

    def test_dataset_length(self):
        expected_length = len(os.listdir(config.PHOTOS_DIR))
        self.assertEqual(len(self.dataset), expected_length)

    def test_if_photos_and_monet_images_have_equal_shape(self):
        self.assertEqual(self.photo.shape, self.monet.shape)

    def test_dataset_images_shape(self):
        expected_shape = (config.IMAGE_CHANNELS, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
        self.assertEqual(self.photo.shape, expected_shape)

    def test_if_dataset_images_in_correct_value_range(self):
        is_image_greater_than_negative_one = torch.all(self.photo >= -1.0)
        is_image_smaller_than_one = torch.all(self.photo <= 1)
        is_image_in_range = torch.stack([is_image_greater_than_negative_one,
                                         is_image_smaller_than_one])

        self.assertTrue(torch.all(is_image_in_range))

    def test_if_dataset_images_are_tensors(self):
        self.assertIsInstance(self.photo, torch.Tensor)
