import os
from os.path import expanduser

import torch

from deeplite_torch_zoo.src.classification.datasets.imagenette import Imagenette
from deeplite_torch_zoo.src.classification.augmentations.augs import (
    get_imagenet_transforms,
    get_vanilla_transforms,
)
from deeplite_torch_zoo.api.datasets.utils import create_loader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY
from deeplite_torch_zoo.src.classification.augmentations.augs import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT


__all__ = ['get_imagenette', 'get_imagenette_320', 'get_imagenette_160']

IMAGENETTE_IMAGENET_CLS_LABEL_MAP = (0, 217, 482, 491, 497, 566, 569, 571, 574, 701)




@DATASET_WRAPPER_REGISTRY.register(dataset_name='imagenette')


@DATASET_WRAPPER_REGISTRY.register(dataset_name='imagenette_320')


@DATASET_WRAPPER_REGISTRY.register(dataset_name='imagenette_160')