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
from ..base_model import MomentumDistilationMixin
from .blip import BlipBase
from .blip_outputs import (
    BlipIntermediateOutput,
    BlipOutputWithLogits,
)
from ..med import XBertEncoder
from ..vit import VisionTransformerEncoder
from torch import nn


@registry.register_model("blip_classification")
class BlipClassification(BlipBase, MomentumDistilationMixin):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_classification_base.yaml",
    }





    @classmethod