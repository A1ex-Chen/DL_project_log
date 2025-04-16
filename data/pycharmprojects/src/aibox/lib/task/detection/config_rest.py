from ast import literal_eval
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Tuple, List, Dict, Any, Union

from torch import Tensor

from .algorithm import Algorithm
from .backbone import Backbone
from .evaluator import Evaluator
from ... import config
from ...config import REQUIRED, LAZY_DEFAULT


@dataclass
class Config(config.Config):

    algorithm_name: Algorithm.Name = REQUIRED
    backbone_name: Backbone.Name = REQUIRED

    anchor_ratios: List[Tuple[int, int]] = LAZY_DEFAULT
    anchor_sizes: List[int] = LAZY_DEFAULT

    backbone_pretrained: bool = True
    backbone_num_frozen_levels: int = 2

    train_rpn_pre_nms_top_n: int = LAZY_DEFAULT
    train_rpn_post_nms_top_n: int = LAZY_DEFAULT

    eval_rpn_pre_nms_top_n: int = LAZY_DEFAULT
    eval_rpn_post_nms_top_n: int = LAZY_DEFAULT

    num_anchor_samples_per_batch: int = 256
    num_proposal_samples_per_batch: int = 128
    num_detections_per_image: int = 100

    anchor_smooth_l1_loss_beta: float = 1.0
    proposal_smooth_l1_loss_beta: float = 1.0

    proposal_nms_threshold: float = 0.7
    detection_nms_threshold: float = 0.5

    eval_quality: str = Evaluator.Evaluation.Quality.STANDARD


    @staticmethod