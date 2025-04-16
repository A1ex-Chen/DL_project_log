# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # noqa . for compatibility




# Note: this function (nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future


# Note: this function (batched_nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future