import os
from typing import List

import config
import matplotlib.pyplot as plt


def save_epoch_loss_results(epoch: int, losses: List[List[float]]):
    generator_losses, discriminator_losses = losses
    steps = [i for i, _ in enumerate(generator_losses)]

    # plot loss for each model
    plt.plot(steps, generator_losses, label='Generator loss')
    plt.plot(steps, discriminator_losses, label='Discriminator loss')
    plt.grid(True)

    # set plot labels and title
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Generator and discriminator loss in epoch {epoch+1}')

    plt.savefig(os.path.join(config.LOSS_PLOTS_DIR, f'{epoch+1}.png'))

    return steps
