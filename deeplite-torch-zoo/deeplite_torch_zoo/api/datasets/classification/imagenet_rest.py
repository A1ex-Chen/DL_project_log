import torch
from timm.data import create_dataset

from deeplite_torch_zoo.src.classification.augmentations.augs import (
    get_imagenet_transforms,
)
from deeplite_torch_zoo.api.datasets.utils import create_loader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY
from deeplite_torch_zoo.src.classification.augmentations.augs import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT


__all__ = ['get_imagenet']


@DATASET_WRAPPER_REGISTRY.register(dataset_name='imagenet16')
@DATASET_WRAPPER_REGISTRY.register(dataset_name='imagenet10')
@DATASET_WRAPPER_REGISTRY.register(dataset_name='imagenet')