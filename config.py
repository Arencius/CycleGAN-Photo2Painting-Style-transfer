import os
import torch
from torchvision.transforms import transforms


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_CHANNELS = 3
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
LAMBDA_CYCLE = 10
BATCH_SIZE = 1
ALPHA = 0.2
BETA_1 = 0.5
BETA_2 = 0.999
LEARNING_RATE = 1e-5
EPOCHS = 15
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(ROOT_DIR, 'src/dataset/monet_jpg/')
MONET_DIR = os.path.join(ROOT_DIR, 'src/dataset/monet_jpg/')
IMG_TRANSFORMS = transforms.Compose([
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                transforms.RandomHorizontalFlip(p=0.5)
            ])
L1Loss = torch.nn.L1Loss()
MSELoss = torch.nn.MSELoss()
