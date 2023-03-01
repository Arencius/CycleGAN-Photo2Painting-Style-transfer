import os
from typing import List

import config
import torch
import numpy as np
import matplotlib.pyplot as plt


def is_tensor_in_range(tensor, range_min=-1.0, range_max=1.0):
    return torch.stack([torch.logical_and(tensor >= range_min,
                                          tensor <= range_max)]).all()


def denormalize(*args):
    return [arg * 0.5 + 0.5 for arg in args]


def save_epoch_loss_results(epoch: int, losses: List[np.array]):
    generator_losses, discriminator_losses = losses
    steps = [i for i, _ in enumerate(generator_losses)]

    plt.plot(steps, generator_losses, label='Generator loss')
    plt.plot(steps, discriminator_losses, label='Discriminator loss')

    plt.grid(True)

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Generator and discriminator loss in epoch {epoch + 1}')

    plt.savefig(os.path.join(config.LOSS_PLOTS_DIR, f'{epoch + 1}.png'))
    plt.close()
