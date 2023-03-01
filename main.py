import config
from torch.utils.data import DataLoader
from src.dataset.photo2painting_dataset import Photo2PaintingDataset
from src.model.generator import Generator
from src.model.discriminator import Discriminator
from training import train_model


if __name__ == '__main__':
    painting_discriminator = Discriminator().to(config.DEVICE)
    painting_generator = Generator(res_blocks=6).to(config.DEVICE)

    photo_discriminator = Discriminator().to(config.DEVICE)
    photo_generator = Generator(res_blocks=6).to(config.DEVICE)

    dataset = Photo2PaintingDataset()
    data_loader = DataLoader(dataset,
                             batch_size=config.BATCH_SIZE,
                             shuffle=True)

    train_model(painting_discriminator,
                painting_generator,
                photo_discriminator,
                photo_generator,
                data_loader)
