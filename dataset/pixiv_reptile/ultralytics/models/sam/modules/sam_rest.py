# Ultralytics YOLO 🚀, AGPL-3.0 license

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch import nn

from .decoders import MaskDecoder
from .encoders import ImageEncoderViT, PromptEncoder


class Sam(nn.Module):
    """
    Sam (Segment Anything Model) is designed for object segmentation tasks. It uses image encoders to generate image
    embeddings, and prompt encoders to encode various types of input prompts. These embeddings are then used by the mask
    decoder to predict object masks.

    Attributes:
        mask_threshold (float): Threshold value for mask prediction.
        image_format (str): Format of the input image, default is 'RGB'.
        image_encoder (ImageEncoderViT): The backbone used to encode the image into embeddings.
        prompt_encoder (PromptEncoder): Encodes various types of input prompts.
        mask_decoder (MaskDecoder): Predicts object masks from the image and prompt embeddings.
        pixel_mean (List[float]): Mean pixel values for image normalization.
        pixel_std (List[float]): Standard deviation values for image normalization.
    """

    mask_threshold: float = 0.0
    image_format: str = "RGB"
