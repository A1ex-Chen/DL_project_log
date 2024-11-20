import os
from os.path import expanduser

import torch
from torchvision import datasets

from deeplite_torch_zoo.src.classification.augmentations.augs import (
    get_vanilla_transforms, get_imagenet_transforms
)
from deeplite_torch_zoo.api.datasets.utils import create_loader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY

__all__ = ['get_tinyimagenet']

TINYIMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINYIMAGENET_STD = (0.2302, 0.2265, 0.2262)


@DATASET_WRAPPER_REGISTRY.register(dataset_name='tinyimagenet')