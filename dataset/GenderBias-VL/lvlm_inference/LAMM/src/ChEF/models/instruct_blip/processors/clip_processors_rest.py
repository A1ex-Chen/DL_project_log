"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from ..common.registry import registry
from .blip_processors import BlipImageBaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode




@registry.register_processor("clip_image_train")
class ClipImageTrainProcessor(BlipImageBaseProcessor):

    @classmethod


@registry.register_processor("clip_image_eval")
class ClipImageEvalProcessor(BlipImageBaseProcessor):

    @classmethod