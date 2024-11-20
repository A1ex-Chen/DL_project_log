# Ultralytics YOLO 🚀, AGPL-3.0 license

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch

from ultralytics.utils.downloads import attempt_download_asset

from .modules.decoders import MaskDecoder
from .modules.encoders import ImageEncoderViT, PromptEncoder
from .modules.sam import Sam
from .modules.tiny_encoder import TinyViT
from .modules.transformer import TwoWayTransformer












sam_model_map = {
    "sam_h.pt": build_sam_vit_h,
    "sam_l.pt": build_sam_vit_l,
    "sam_b.pt": build_sam_vit_b,
    "mobile_sam.pt": build_mobile_sam,
}

