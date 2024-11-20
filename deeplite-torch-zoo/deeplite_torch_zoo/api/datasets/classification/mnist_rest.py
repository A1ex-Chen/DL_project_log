import os
from os.path import expanduser

import torch
import torchvision
from torchvision import transforms

from deeplite_torch_zoo.api.datasets.utils import create_loader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY


__all__ = ["get_mnist"]

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


@DATASET_WRAPPER_REGISTRY.register(dataset_name='mnist')