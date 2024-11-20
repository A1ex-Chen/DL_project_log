"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import warnings

import torch
import torch.nn.functional as F
from ...common.registry import registry
from .blip import BlipBase
from .blip_outputs import BlipOutputFeatures
from ..med import XBertEncoder
from ..vit import VisionTransformerEncoder
from torch import nn


@registry.register_model("blip_feature_extractor")
class BlipFeatureExtractor(BlipBase):
    """
    Class for BLIP feature extractor.

    Supported model types:
        - base: BLIP base model with pre-trained weights from capfilt by BLIP large model.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_feature_extractor", "base")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_feature_extractor_base.yaml",
        # "large": "configs/models/blip_feature_extractor_large.yaml",
    }


    @torch.no_grad()

    @classmethod