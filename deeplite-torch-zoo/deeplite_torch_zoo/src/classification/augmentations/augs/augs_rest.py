from timm.data.transforms_factory import (
    transforms_imagenet_eval,
    transforms_imagenet_train,
)
from timm.data.transforms import ToNumpy
from torchvision import transforms

from deeplite_torch_zoo.src.classification.augmentations.augs.cutout import Cutout


DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)



