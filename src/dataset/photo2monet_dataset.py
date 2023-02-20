import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image

import config


class Photo2MonetDataset(Dataset):
    def __init__(self, transform=True):
        self.transform = transform
        self.photos_data = os.listdir(config.PHOTOS_IMAGES_DIR)
        self.monet_data = os.listdir(config.MONET_IMAGES_DIR)

    def read_image_from_directory(self, category, index):
        categories = {
            'photo': (self.photos_data, config.PHOTOS_IMAGES_DIR),
            'monet': (self.monet_data, config.MONET_IMAGES_DIR),
        }

        files_category, images_dir = categories.get(category)
        image_filename = files_category[index]
        path_to_image = os.path.join(images_dir, image_filename)

        image = Image.open(path_to_image).convert('RGB')

        if self.transform:
            image = config.IMG_TRANSFORMS(image)

        return image

    def __len__(self):
        return min(len(self.photos_data), len(self.monet_data))

    def __getitem__(self, index):
        photo_image = self.read_image_from_directory('photo', index)
        monet_image = self.read_image_from_directory('monet', index)

        return photo_image, monet_image





