from typing import Tuple, Callable

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops.boxes import nms, box_iou

from ....bbox import BBox
from ....extension.functional import beta_smooth_l1_loss


class ROI(nn.Module):




