import os
from os.path import expanduser

import torch
import torchvision
from torchvision import transforms

from deeplite_torch_zoo.api.datasets.utils import create_loader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY


__all__ = ['get_cifar100', 'get_cifar10']


CIFAR_IMAGE_SIZE = 32
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)




@DATASET_WRAPPER_REGISTRY.register(dataset_name='cifar100')


@DATASET_WRAPPER_REGISTRY.register(dataset_name='cifar10')