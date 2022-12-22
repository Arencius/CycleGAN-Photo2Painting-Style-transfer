import os
from torchvision.transforms import transforms

IMAGE_CHANNELS = 3
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
BATCH_SIZE = 1
ALPHA = 0.2
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(ROOT_DIR, 'src/dataset/monet_jpg/')
MONET_DIR = os.path.join(ROOT_DIR, 'src/dataset/monet_jpg/')
IMG_TRANSFORMS = transforms.Compose([
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                transforms.RandomHorizontalFlip(p=0.5)
            ])
