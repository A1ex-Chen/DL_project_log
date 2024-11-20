# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Tuple
import torch
from PIL import Image
from torch.nn import functional as F

__all__ = ["paste_masks_in_image"]


BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit




# Annotate boxes as Tensor (but not Boxes) in order to use scripting
@torch.jit.script_if_tracing


# The below are the original paste function (from Detectron1) which has
# larger quantization error.
# It is faster on CPU, while the aligned one is faster on GPU thanks to grid_sample.




# Our pixel modeling requires extrapolation for any continuous
# coordinate < 0.5 or > length - 0.5. When sampling pixels on the masks,
# we would like this extrapolation to be an interpolation between boundary values and zero,
# instead of using absolute zero or boundary values.
# Therefore `paste_mask_in_image_old` is often used with zero padding around the masks like this:
# masks, scale = pad_masks(masks[:, 0, :, :], 1)
# boxes = scale_boxes(boxes.tensor, scale)






@torch.jit.script_if_tracing