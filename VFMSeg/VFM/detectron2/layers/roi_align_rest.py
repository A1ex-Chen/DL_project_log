# Copyright (c) Facebook, Inc. and its affiliates.
from torch import nn
from torchvision.ops import roi_align


# NOTE: torchvision's RoIAlign has a different default aligned=False
class ROIAlign(nn.Module):

