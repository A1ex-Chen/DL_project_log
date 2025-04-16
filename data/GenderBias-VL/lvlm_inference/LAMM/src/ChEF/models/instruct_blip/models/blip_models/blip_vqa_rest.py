"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn.functional as F
from ...common.registry import registry
from ..base_model import tile
from .blip import BlipBase
from .blip_outputs import (
    BlipOutput,
    BlipIntermediateOutput,
)
from ..med import XBertEncoder, XBertLMHeadDecoder
from ..vit import VisionTransformerEncoder


@registry.register_model("blip_vqa")
class BlipVQA(BlipBase):
    """
    BLIP VQA models.

    Supported model types:
        - base: vqa model initialized with pre-trained BLIP base model on 115M image-text pairs after CapFilt; not fine-tuned.
        - vqav2: fine-tuned BLIP base model on VQA v2.0 dataset.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_vqa", "vqav2")
        >>> model = load_model("blip_vqa", "okvqa")
        >>> model = load_model("blip_vqa", "aokvqa")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vqav2": "configs/models/blip_vqav2.yaml",
        "okvqa": "configs/models/blip_vqa_okvqa.yaml",
        "aokvqa": "configs/models/blip_vqa_aokvqa.yaml",
    }








    @classmethod