# Code modified from: https://github.com/DeepVoltaire/AutoAugment

import numpy as np
import torch


class Cutout:
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

