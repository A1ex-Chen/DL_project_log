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


class BlipImageBaseProcessor(BaseProcessor):


@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor):


    @classmethod



@registry.register_processor("blip_question")
class BlipQuestionProcessor(BaseProcessor):


    @classmethod



@registry.register_processor("blip_image_train")
class BlipImageTrainProcessor(BlipImageBaseProcessor):


    @classmethod


@registry.register_processor("blip_image_eval")
class BlipImageEvalProcessor(BlipImageBaseProcessor):


    @classmethod


@registry.register_processor("blip2_image_train")
class Blip2ImageTrainProcessor(BlipImageBaseProcessor):


    @classmethod