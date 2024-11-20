import os
from os.path import expanduser

import torch

from deeplite_torch_zoo.src.classification.datasets.imagewoof import Imagewoof
from deeplite_torch_zoo.src.classification.augmentations.augs import (
    get_imagenet_transforms,
    get_vanilla_transforms,
)
from deeplite_torch_zoo.api.datasets.utils import create_loader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY
from deeplite_torch_zoo.src.classification.augmentations.augs import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT


__all__ = ['get_imagewoof', 'get_imagewoof_320', 'get_imagewoof_160']




@DATASET_WRAPPER_REGISTRY.register(dataset_name='imagewoof')


@DATASET_WRAPPER_REGISTRY.register(dataset_name='imagewoof_320')


@DATASET_WRAPPER_REGISTRY.register(dataset_name='imagewoof_160')