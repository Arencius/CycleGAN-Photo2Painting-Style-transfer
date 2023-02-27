import config
import utils
import torch
import torchvision
from tqdm import tqdm


def train_model(monet_discriminator,
                monet_gen,
                photo_discriminator,
                photo_generator,
                data_loader):
    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    discriminator_optimizer = torch.optim.Adam(
        list(photo_discriminator.parameters()) + list(monet_discriminator.parameters()),
        lr=config.DISCRIMINATOR_LEARNING_RATE,
        betas=(config.BETA_1, config.BETA_2)
    )
    generator_optimizer = torch.optim.Adam(
        list(monet_gen.parameters()) + list(photo_generator.parameters()),
        lr=config.GENERATOR_LEARNING_RATE,
        betas=(config.BETA_1, config.BETA_2)
    )

    print(f'Training on: {config.DEVICE}\n')

    for epoch in range(config.EPOCHS):
        print(f'Epoch: {epoch + 1}')
        generator_losses, discriminator_losses = [], []

        if 0 < epoch < 10:
            discriminator_optimizer.param_groups[0]['lr'] *= 0.5
            generator_optimizer.param_groups[0]['lr'] *= 0.5

        for batch_index, (photo, monet) in enumerate(tqdm(data_loader)):
            photo = photo.to(config.DEVICE)
            monet = monet.to(config.DEVICE)

            # train discriminators
            with torch.cuda.amp.autocast():
                # photo discriminator
                fake_photo = photo_generator(monet)
                disc_photo_real = photo_discriminator(photo)
                photo_discriminator_output = photo_discriminator(fake_photo.detach())

                photo_disc_real_loss = config.MSELoss(disc_photo_real, torch.ones_like(disc_photo_real))
                photo_disc_fake_loss = config.MSELoss(photo_discriminator_output, torch.zeros_like(photo_discriminator_output))
                photo_discriminator_loss = photo_disc_real_loss + photo_disc_fake_loss

                # monet discriminator
                fake_monet = monet_gen(photo)
                disc_monet_real = monet_discriminator(monet)
                monet_discriminator_output = monet_discriminator(fake_monet.detach())

                monet_disc_real_loss = config.MSELoss(disc_monet_real, torch.ones_like(disc_monet_real))
                monet_disc_fake_loss = config.MSELoss(monet_discriminator_output, torch.zeros_like(monet_discriminator_output))
                monet_discriminator_loss = monet_disc_real_loss + monet_disc_fake_loss

                discriminator_loss = (photo_discriminator_loss + monet_discriminator_loss)

            # update discriminator optimizer
            discriminator_optimizer.zero_grad()
            scaler_d.scale(discriminator_loss).backward()
            scaler_d.step(discriminator_optimizer)
            scaler_d.update()

            # train generators
            with torch.cuda.amp.autocast():
                # adversarial loss
                photo_discriminator_output = photo_discriminator(fake_photo)
                monet_discriminator_output = monet_discriminator(fake_monet)

                fake_photo_logits = torch.ones_like(photo_discriminator_output) * 0.9
                fake_monet_logits = torch.ones_like(monet_discriminator_output) * 0.9

                adversarial_photo_loss = config.MSELoss(photo_discriminator_output, fake_photo_logits)
                adversarial_monet_loss = config.MSELoss(monet_discriminator_output, fake_monet_logits)

                # cycle loss
                cycle_photo_loss = config.L1Loss(photo, photo_generator(fake_monet))
                cycle_monet_loss = config.L1Loss(monet, monet_gen(fake_photo))

                # identity_loss
                identity_photo_loss = config.L1Loss(photo, photo_generator(photo))
                identity_monet_loss = config.L1Loss(monet, monet_gen(monet))

                # combined loss
                generator_loss = 0.75 * (
                        adversarial_photo_loss
                        + adversarial_monet_loss
                        + cycle_photo_loss * config.LAMBDA_CYCLE
                        + cycle_monet_loss * config.LAMBDA_CYCLE
                        + identity_photo_loss * config.LAMBDA_IDENTITY
                        + identity_monet_loss * config.LAMBDA_IDENTITY
                )

            # update generator optimizer
            generator_optimizer.zero_grad()
            scaler_g.scale(generator_loss).backward()
            scaler_g.step(generator_optimizer)
            scaler_g.update()

            generator_losses.append(generator_loss.item())
            discriminator_losses.append(discriminator_loss.item())

            if batch_index % config.VALIDATION_STEP == 0:
                real_monet_painting, generated_photo, \
                    real_photo, generated_monet_painting = utils.denormalize(monet, fake_photo, photo, fake_monet)

                torchvision.utils.save_image(torch.cat([real_monet_painting, generated_photo], dim=3),
                                             f"{config.PHOTOS_RESULTS_DIR}/generated_photo_epoch{epoch + 1}_{batch_index}.png")
                torchvision.utils.save_image(torch.cat([real_photo, generated_monet_painting], dim=3),
                                             f"{config.MONET_RESULTS_DIR}/generated_monet_epoch{epoch + 1}_{batch_index}.png")

        print(f'Mean generator loss: {torch.tensor(generator_loss, dtype=torch.float32).mean()}\n'
              f'Mean discriminator loss: {torch.tensor(discriminator_loss, dtype=torch.float32).mean()}\n')

        utils.save_epoch_loss_results(epoch, [generator_losses,
                                              discriminator_losses])
