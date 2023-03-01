import os

from torch.utils.data import Dataset
from PIL import Image

import config


class Photo2PaintingDataset(Dataset):
    def __init__(self, transform=True):
        self.transform = transform
        self.photos_data = os.listdir(config.PHOTOS_IMAGES_DIR)
        self.paintings_data = os.listdir(config.PAINTINGS_IMAGES_DIR)

    def read_image_from_directory(self, category, index):
        images_categories = {
            'photo': (self.photos_data, config.PHOTOS_IMAGES_DIR),
            'painting': (self.paintings_data, config.PAINTINGS_IMAGES_DIR),
        }

        files_category, images_dir = images_categories.get(category)
        image_filename = files_category[index]
        path_to_image = os.path.join(images_dir, image_filename)

        image = Image.open(path_to_image).convert('RGB')

        if self.transform:
            image = config.IMG_TRANSFORMS(image)

        return image

    def __len__(self):
        return min(len(self.photos_data), len(self.paintings_data))

    def __getitem__(self, index):
        photo_image = self.read_image_from_directory('photo', index)
        painting_image = self.read_image_from_directory('painting', index)

        return photo_image, painting_image





