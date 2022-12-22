import os
from torchvision.transforms import transforms

IMAGE_CHANNELS = 3
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
BATCH_SIZE = 4
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(ROOT_DIR, 'src/dataset/monet_jpg/')
MONET_DIR = os.path.join(ROOT_DIR, 'src/dataset/monet_jpg/')
IMG_TRANSFORMS = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
