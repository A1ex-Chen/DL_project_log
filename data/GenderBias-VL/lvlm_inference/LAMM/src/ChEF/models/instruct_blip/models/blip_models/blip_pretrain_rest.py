"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from ...common.registry import registry
from ..base_model import MomentumDistilationMixin, SharedQueueMixin
from . import tie_encoder_decoder_weights
from .blip import BlipBase
from .blip_outputs import (
    BlipOutput,
    BlipSimilarity,
    BlipIntermediateOutput,
)
from ..med import XBertEncoder, XBertLMHeadDecoder
from ..vit import VisionTransformerEncoder
from torch import nn


@registry.register_model("blip_pretrain")
class BlipPretrain(BlipBase, SharedQueueMixin, MomentumDistilationMixin):
    """
    BLIP pretrain model.

    Supported model types:
        - base: BLIP base model before pretraining.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_pretrain_base.yaml",
        # "large": "configs/models/blip_pretrain_large.yaml",
    }





    @classmethod