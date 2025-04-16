import pickle
from ast import literal_eval
from dataclasses import dataclass, asdict
from distutils.util import strtobool
from typing import List, Optional, Tuple, Dict, Any, Union

from torch import Tensor

import aibox
from .augmenter import Augmenter
from .task import Task


class RequiredType:
    pass


class LazyDefaultType:
    pass


REQUIRED = RequiredType()
LAZY_DEFAULT = LazyDefaultType()


@dataclass
class Config:

    task_name: Task.Name = REQUIRED

    path_to_checkpoints_dir: str = REQUIRED
    path_to_data_dir: str = REQUIRED
    path_to_extra_data_dirs: Tuple[str, ...] = ()

    path_to_resuming_checkpoint: Optional[str] = None
    path_to_finetuning_checkpoint: Optional[str] = None
    path_to_loading_checkpoint: Optional[str] = None

    num_workers: int = 2
    visible_devices: Optional[List[int]] = None

    needs_freeze_bn: bool = LAZY_DEFAULT

    image_resized_width: int = LAZY_DEFAULT
    image_resized_height: int = LAZY_DEFAULT
    image_min_side: int = LAZY_DEFAULT
    image_max_side: int = LAZY_DEFAULT
    image_side_divisor: int = LAZY_DEFAULT

    aug_strategy: Augmenter.Strategy = Augmenter.Strategy.ALL
    aug_hflip_prob: float = 0.5
    aug_vflip_prob: float = 0
    aug_rotate90_prob: float = 0
    aug_crop_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (0, 1))
    aug_zoom_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_scale_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_translate_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_rotate_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_shear_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_blur_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (0, 1))
    aug_sharpen_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (0, 1))
    aug_color_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_brightness_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_grayscale_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (0, 1))
    aug_contrast_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_noise_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (0, 1))
    aug_resized_crop_prob_and_width_height: Tuple[float, Tuple[int, int]] = (0, (224, 224))

    batch_size: int = LAZY_DEFAULT
    learning_rate: float = LAZY_DEFAULT
    momentum: float = 0.9
    weight_decay: float = 0.0005
    clip_grad_base_and_max: Optional[str] = None
    step_lr_sizes: List[int] = LAZY_DEFAULT
    step_lr_gamma: float = 0.1
    warm_up_factor: float = 0.3333
    warm_up_num_iters: int = 500

    num_batches_to_display: int = 20
    num_epochs_to_validate: int = 1
    num_epochs_to_finish: int = LAZY_DEFAULT
    max_num_checkpoints: int = 6

    @staticmethod




    @staticmethod