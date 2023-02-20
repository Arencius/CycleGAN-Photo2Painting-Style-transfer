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
            'photo': self.photos_data,
            'monet': self.monet_data
        }

        image_filename = categories.get(category)[index]
        path_to_image = os.path.join(config.PHOTOS_IMAGES_DIR, image_filename)
        image = Image.open(path_to_image).convert('RGB')

        return image

    def __len__(self):
        return min(len(self.photos_data), len(self.monet_data))

    def __getitem__(self, index):
        '''
                photo_image = self.read_image_from_directory('photo', index)
                monet_image = self.read_image_from_directory('monet', index)
                return photo_image, monet_image
        '''
        photo_filename = self.photos_data[index]
        monet_filename = self.monet_data[index]

        path_to_photo = os.path.join(config.PHOTOS_IMAGES_DIR, photo_filename)
        path_to_monet = os.path.join(config.MONET_IMAGES_DIR, monet_filename)

        photo_image = Image.open(path_to_photo).convert('RGB')
        monet_image = Image.open(path_to_monet).convert('RGB')

        if self.transform:
            photo_image = config.IMG_TRANSFORMS(photo_image)
            monet_image = config.IMG_TRANSFORMS(monet_image)

        return photo_image, monet_image

