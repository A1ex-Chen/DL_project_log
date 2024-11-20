"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os

import torch
import torch.nn.functional as F
from ...common.dist_utils import download_cached_file
from ...common.registry import registry
from ...common.utils import get_abs_path, is_url
from ..base_model import MomentumDistilationMixin
from ..blip_models.blip import BlipBase
from .blip_outputs import BlipIntermediateOutput, BlipOutput
from .nlvr_encoder import BertModel
from ..vit import VisionTransformerEncoder, interpolate_pos_embed
from torch import nn
from transformers import BertConfig


@registry.register_model("blip_nlvr")
class BlipNLVR(BlipBase, MomentumDistilationMixin):
    """
    Class for BLIP NLVR model.

    Supported model types:
        - base: model with pre-trained BLIP weights, used as initialization for fine-tuning.
        - nlvr: finetuned model on NLVR2 dataset.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_nlvr", "nlvr")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "nlvr": "configs/models/blip_nlvr.yaml",
    }




    @classmethod
