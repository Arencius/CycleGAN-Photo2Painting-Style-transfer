import os
import torch
from torchvision.transforms import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DISCRIMINATOR_PATCH_SIZE = 65
IMAGE_CHANNELS = 3
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = LAMBDA_CYCLE * 0.5
BATCH_SIZE = 2
ALPHA = 0.2
BETA_1 = 0.5
BETA_2 = 0.999
GENERATOR_LEARNING_RATE = 2e-4
DISCRIMINATOR_LEARNING_RATE = GENERATOR_LEARNING_RATE * 0.5
EPOCHS = 15
VALIDATION_STEP = 100
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_IMAGES_DIR = os.path.join(ROOT_DIR, 'src/dataset/photo_jpg/')
MONET_IMAGES_DIR = os.path.join(ROOT_DIR, 'src/dataset/monet_jpg/')
PHOTOS_RESULTS_DIR = os.path.join(ROOT_DIR, 'training_results/Generated photos/')
MONET_RESULTS_DIR = os.path.join(ROOT_DIR, 'training_results/Generated paintings/')
LOSS_PLOTS_DIR = os.path.join(ROOT_DIR, 'training_results/Models loss/')

IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT * 2, IMAGE_WIDTH * 2)),
    transforms.CenterCrop(IMAGE_WIDTH),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
L1Loss = torch.nn.L1Loss()
MSELoss = torch.nn.MSELoss()
