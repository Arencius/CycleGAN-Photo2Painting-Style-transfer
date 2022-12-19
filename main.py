from torchsummary import summary
from src.model.generator import Generator


if __name__ == '__main__':
    generator = Generator().cuda()

    summary(generator, (3, 256, 256))
