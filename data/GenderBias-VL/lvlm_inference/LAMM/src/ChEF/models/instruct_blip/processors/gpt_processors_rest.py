"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from ..common.registry import registry
from .base_processor import BaseProcessor
from .randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
from itertools import chain
import numpy as np
import torch
from transformers import GPT2Tokenizer

SPECIAL_TOKENS_DICT = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<video>", "<cap>"],
    "pad_token": "<pad>",
}
SPECIAL_TOKENS = [
    "<bos>",
    "<eos>",
    "<speaker1>",
    "<speaker2>",
    "<cap>",
    "<video>",
    "<pad>",
]


class GPTVideoFeatureBaseProcessor(BaseProcessor):


@registry.register_processor("gpt_dialogue")
class GPTDialogueProcessor(BaseProcessor):





    @classmethod


@registry.register_processor("gpt_video_ft")
class GPTVideoFeatureProcessor(GPTVideoFeatureBaseProcessor):




    @classmethod