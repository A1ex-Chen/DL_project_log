from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor
from torchvision.ops.boxes import remove_small_boxes

from .model import Model
from ...extension.data_parallel import BunchDataParallel, Bunch


class Inferer:

    @dataclass
    class Inference:
        anchor_bboxes_batch: List[Tensor]
        proposal_bboxes_batch: List[Tensor]
        proposal_probs_batch: List[Tensor]
        detection_bboxes_batch: List[Tensor]
        detection_classes_batch: List[Tensor]
        detection_probs_batch: List[Tensor]
        final_detection_bboxes_batch: List[Tensor]
        final_detection_classes_batch: List[Tensor]
        final_detection_probs_batch: List[Tensor]


    @torch.no_grad()