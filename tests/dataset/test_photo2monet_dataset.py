import os
import unittest
import random
import config

from src.dataset.photo2monet_dataset import Photo2MonetDataset


class TestPhoto2MonetDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Photo2MonetDataset()
        self.random_index = random.randint(0, len(self.dataset))

    def test_dataset_length(self):
        expected_length = len(os.listdir(config.PHOTOS_DIR))
        self.assertEqual(len(self.dataset), expected_length)

    def test_if_photos_and_monet_images_have_equal_shape(self):
        photo, monet = self.dataset[self.random_index]
        self.assertEqual(photo.shape, monet.shape)

    def test_dataset_images_shape(self):
        photo, _ = self.dataset[self.random_index]
        expected_shape = (config.IMAGE_CHANNELS, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
        self.assertEqual(photo.shape, expected_shape)

    def test_dataset_images_value_range(self):
        pass
        #expected_min, expected_max = -1.0, 1.0

