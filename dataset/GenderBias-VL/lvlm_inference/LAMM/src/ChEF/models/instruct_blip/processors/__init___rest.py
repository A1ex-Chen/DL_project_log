"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from .base_processor import BaseProcessor

from .blip_processors import (
    BlipImageTrainProcessor,
    Blip2ImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)
from .gpt_processors import (
    GPTVideoFeatureProcessor,
    GPTDialogueProcessor,
)
from.clip_processors import ClipImageTrainProcessor

from ..common.registry import registry

__all__ = [
    "BaseProcessor",
    # BLIP
    "BlipImageTrainProcessor",
    "Blip2ImageTrainProcessor",
    "BlipImageEvalProcessor",
    "BlipCaptionProcessor",
    "ClipImageTrainProcessor",
    # GPT
    "GPTVideoFeatureProcessor",
    "GPTDialogueProcessor",
]

