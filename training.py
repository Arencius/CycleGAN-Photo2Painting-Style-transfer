import config
import torch
from tqdm import tqdm


def train_model(monet_disc,
                monet_gen,
                photo_disc,
                photo_gen,
                data_loader):
    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    discriminator_opt = torch.optim.Adam(
        list(monet_disc.parameters()) + list(photo_disc.parameters()),
        lr=config.LEARNING_RATE,
        betas=(config.BETA_1, config.BETA_2)
    )
    generator_opt = torch.optim.Adam(
        list(monet_gen.parameters()) + list(photo_gen.parameters()),
        lr=config.LEARNING_RATE,
        betas=(config.BETA_1, config.BETA_2)
    )

    for epoch in range(config.EPOCHS):
        for index, (photo, monet) in enumerate(tqdm(data_loader)):
            photo = photo.to(config.DEVICE)
            monet = monet.to(config.DEVICE)

            # train discriminators
            with torch.cuda.amp.autocast():
                # photo discriminator
                fake_photo = photo_gen(monet)
                disc_photo_real = photo_disc(photo)
                disc_photo_fake = photo_disc(fake_photo.detach())

                photo_disc_real_loss = config.MSELoss(disc_photo_real, torch.ones_like(disc_photo_real))
                photo_disc_fake_loss = config.MSELoss(disc_photo_fake, torch.zeros_like(disc_photo_fake))
                photo_discriminator_loss = photo_disc_real_loss + photo_disc_fake_loss

                # monet discriminator
                fake_monet = monet_gen(photo)
                disc_monet_real = monet_disc(monet)
                disc_monet_fake = monet_disc(fake_monet.detach())

                monet_disc_real_loss = config.MSELoss(disc_monet_real, torch.ones_like(disc_monet_real))
                monet_disc_fake_loss = config.MSELoss(disc_monet_fake, torch.zeros_like(disc_monet_fake))
                monet_discriminator_loss = monet_disc_real_loss + monet_disc_fake_loss

                discriminator_loss = 0.5 * (photo_discriminator_loss + monet_discriminator_loss)

            # update discriminator optimizer
            discriminator_opt.zero_grad()
            scaler_d.scale(discriminator_loss).backward()
            scaler_d.step(discriminator_opt)
            scaler_d.update()

            # train generators
            with torch.cuda.amp.autocast():
                # adversarial loss
                disc_photo_fake = photo_disc(fake_photo)
                disc_monet_fake = monet_disc(fake_monet)

                adversarial_photo_loss = config.MSELoss(disc_photo_fake, torch.ones_like(disc_photo_fake))
                adversarial_monet_loss = config.MSELoss(disc_monet_fake, torch.ones_like(disc_monet_fake))

                # cycle loss
                cycle_photo_loss = config.L1Loss(photo, photo_gen(fake_monet))
                cycle_monet_loss = config.L1Loss(monet, monet_gen(fake_photo))

                # combined loss
                generator_loss = (
                        adversarial_photo_loss
                        + adversarial_monet_loss
                        + cycle_photo_loss * config.LAMBDA_CYCLE
                        + cycle_monet_loss * config.LAMBDA_CYCLE
                )

            # update generator optimizer
            generator_opt.zero_grad()
            scaler_g.scale(generator_loss).backward()
            scaler_g.step(generator_opt)
            scaler_g.update()

        print(f'Epoch: {epoch+1}\n'
              f'Generator loss: {generator_loss}\n'
              f'Discriminator loss: {discriminator_loss}\n')
