import torch
import torch.nn as nn
import torch.nn.functional as F

from ..general import xywh2xyxy
from ..loss import FocalLoss, smooth_BCE
from ..metrics import bbox_iou
from ..torch_utils import de_parallel
from .general import crop_mask


class ComputeLoss:
    # Compute losses


