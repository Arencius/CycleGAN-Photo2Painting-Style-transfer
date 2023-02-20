import config
from torch.utils.data import DataLoader
from src.dataset.photo2monet_dataset import Photo2MonetDataset
from src.model.generator import Generator
from src.model.discriminator import Discriminator
from training import train_model
from torchsummary import summary


if __name__ == '__main__':
    monet_discriminator = Discriminator().to(config.DEVICE)
    monet_generator = Generator(res_blocks=6).to(config.DEVICE)

    summary(monet_generator, (3,256,256))

    photo_discriminator = Discriminator().to(config.DEVICE)
    photo_generator = Generator(res_blocks=6).to(config.DEVICE)

    dataset = Photo2MonetDataset()
    data_loader = DataLoader(dataset,
                             batch_size=config.BATCH_SIZE,
                             shuffle=True)

    train_model(monet_discriminator,
                monet_generator,
                photo_discriminator,
                photo_generator,
                data_loader)