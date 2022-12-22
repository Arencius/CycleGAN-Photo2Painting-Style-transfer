from torchsummary import summary
from src.model.generator import Generator
from src.model.discriminator import Discriminator


if __name__ == '__main__':
    discriminator = Discriminator().cuda()
    generator = Generator(res_blocks=6).cuda()

    summary(discriminator, (3, 256, 256))
