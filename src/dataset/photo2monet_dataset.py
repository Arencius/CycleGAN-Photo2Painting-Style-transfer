import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image

import config


class Photo2MonetDataset(Dataset):
    def __init__(self, transform=True):
        self.transform = transform
        self.photos_data = os.listdir(config.PHOTOS_DIR)
        self.monet_data = os.listdir(config.MONET_DIR)

    def __len__(self):
        return max(len(self.photos_data), len(self.monet_data))

    def __getitem__(self, index):
        photo_filename = self.photos_data[index]
        monet_filename = self.monet_data[index % len(self.monet_data)]

        path_to_photo = os.path.join(config.PHOTOS_DIR, photo_filename)
        path_to_monet = os.path.join(config.MONET_DIR, monet_filename)

        photo_image = pil_to_tensor(Image.open(path_to_photo).convert('RGB')).type(torch.float)
        monet_image = pil_to_tensor(Image.open(path_to_monet).convert('RGB')).type(torch.float)

        print(type(photo_image))

        if self.transform:
            photo_image = config.IMG_TRANSFORMS(photo_image)
            monet_image = config.IMG_TRANSFORMS(monet_image)

        return photo_image, monet_image
