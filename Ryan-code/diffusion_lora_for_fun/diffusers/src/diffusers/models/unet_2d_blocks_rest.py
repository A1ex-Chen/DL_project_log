# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from ..utils import deprecate
from .unets.unet_2d_blocks import (
    AttnDownBlock2D,
    AttnDownEncoderBlock2D,
    AttnSkipDownBlock2D,
    AttnSkipUpBlock2D,
    AttnUpBlock2D,
    AttnUpDecoderBlock2D,
    AutoencoderTinyBlock,
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    KAttentionBlock,
    KCrossAttnDownBlock2D,
    KCrossAttnUpBlock2D,
    KDownBlock2D,
    KUpBlock2D,
    ResnetDownsampleBlock2D,
    ResnetUpsampleBlock2D,
    SimpleCrossAttnDownBlock2D,
    SimpleCrossAttnUpBlock2D,
    SkipDownBlock2D,
    SkipUpBlock2D,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    UpBlock2D,
    UpDecoderBlock2D,
)








class AutoencoderTinyBlock(AutoencoderTinyBlock):
    deprecation_message = "Importing `AutoencoderTinyBlock` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AutoencoderTinyBlock`, instead."
    deprecate("AutoencoderTinyBlock", "0.29", deprecation_message)


class UNetMidBlock2D(UNetMidBlock2D):
    deprecation_message = "Importing `UNetMidBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D`, instead."
    deprecate("UNetMidBlock2D", "0.29", deprecation_message)


class UNetMidBlock2DCrossAttn(UNetMidBlock2DCrossAttn):
    deprecation_message = "Importing `UNetMidBlock2DCrossAttn` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn`, instead."
    deprecate("UNetMidBlock2DCrossAttn", "0.29", deprecation_message)


class UNetMidBlock2DSimpleCrossAttn(UNetMidBlock2DSimpleCrossAttn):
    deprecation_message = "Importing `UNetMidBlock2DSimpleCrossAttn` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DSimpleCrossAttn`, instead."
    deprecate("UNetMidBlock2DSimpleCrossAttn", "0.29", deprecation_message)


class AttnDownBlock2D(AttnDownBlock2D):
    deprecation_message = "Importing `AttnDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnDownBlock2D`, instead."
    deprecate("AttnDownBlock2D", "0.29", deprecation_message)


class CrossAttnDownBlock2D(CrossAttnDownBlock2D):
    deprecation_message = "Importing `AttnDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D`, instead."
    deprecate("CrossAttnDownBlock2D", "0.29", deprecation_message)


class DownBlock2D(DownBlock2D):
    deprecation_message = "Importing `DownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import DownBlock2D`, instead."
    deprecate("DownBlock2D", "0.29", deprecation_message)


class AttnDownEncoderBlock2D(AttnDownEncoderBlock2D):
    deprecation_message = "Importing `AttnDownEncoderBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnDownEncoderBlock2D`, instead."
    deprecate("AttnDownEncoderBlock2D", "0.29", deprecation_message)


class AttnSkipDownBlock2D(AttnSkipDownBlock2D):
    deprecation_message = "Importing `AttnSkipDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnSkipDownBlock2D`, instead."
    deprecate("AttnSkipDownBlock2D", "0.29", deprecation_message)


class SkipDownBlock2D(SkipDownBlock2D):
    deprecation_message = "Importing `SkipDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import SkipDownBlock2D`, instead."
    deprecate("SkipDownBlock2D", "0.29", deprecation_message)


class ResnetDownsampleBlock2D(ResnetDownsampleBlock2D):
    deprecation_message = "Importing `ResnetDownsampleBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import ResnetDownsampleBlock2D`, instead."
    deprecate("ResnetDownsampleBlock2D", "0.29", deprecation_message)


class SimpleCrossAttnDownBlock2D(SimpleCrossAttnDownBlock2D):
    deprecation_message = "Importing `SimpleCrossAttnDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import SimpleCrossAttnDownBlock2D`, instead."
    deprecate("SimpleCrossAttnDownBlock2D", "0.29", deprecation_message)


class KDownBlock2D(KDownBlock2D):
    deprecation_message = "Importing `KDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KDownBlock2D`, instead."
    deprecate("KDownBlock2D", "0.29", deprecation_message)


class KCrossAttnDownBlock2D(KCrossAttnDownBlock2D):
    deprecation_message = "Importing `KCrossAttnDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KCrossAttnDownBlock2D`, instead."
    deprecate("KCrossAttnDownBlock2D", "0.29", deprecation_message)


class AttnUpBlock2D(AttnUpBlock2D):
    deprecation_message = "Importing `AttnUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnUpBlock2D`, instead."
    deprecate("AttnUpBlock2D", "0.29", deprecation_message)


class CrossAttnUpBlock2D(CrossAttnUpBlock2D):
    deprecation_message = "Importing `CrossAttnUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D`, instead."
    deprecate("CrossAttnUpBlock2D", "0.29", deprecation_message)


class UpBlock2D(UpBlock2D):
    deprecation_message = "Importing `UpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UpBlock2D`, instead."
    deprecate("UpBlock2D", "0.29", deprecation_message)


class UpDecoderBlock2D(UpDecoderBlock2D):
    deprecation_message = "Importing `UpDecoderBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D`, instead."
    deprecate("UpDecoderBlock2D", "0.29", deprecation_message)


class AttnUpDecoderBlock2D(AttnUpDecoderBlock2D):
    deprecation_message = "Importing `AttnUpDecoderBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnUpDecoderBlock2D`, instead."
    deprecate("AttnUpDecoderBlock2D", "0.29", deprecation_message)


class AttnSkipUpBlock2D(AttnSkipUpBlock2D):
    deprecation_message = "Importing `AttnSkipUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnSkipUpBlock2D`, instead."
    deprecate("AttnSkipUpBlock2D", "0.29", deprecation_message)


class SkipUpBlock2D(SkipUpBlock2D):
    deprecation_message = "Importing `SkipUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import SkipUpBlock2D`, instead."
    deprecate("SkipUpBlock2D", "0.29", deprecation_message)


class ResnetUpsampleBlock2D(ResnetUpsampleBlock2D):
    deprecation_message = "Importing `ResnetUpsampleBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import ResnetUpsampleBlock2D`, instead."
    deprecate("ResnetUpsampleBlock2D", "0.29", deprecation_message)


class SimpleCrossAttnUpBlock2D(SimpleCrossAttnUpBlock2D):
    deprecation_message = "Importing `SimpleCrossAttnUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import SimpleCrossAttnUpBlock2D`, instead."
    deprecate("SimpleCrossAttnUpBlock2D", "0.29", deprecation_message)


class KUpBlock2D(KUpBlock2D):
    deprecation_message = "Importing `KUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KUpBlock2D`, instead."
    deprecate("KUpBlock2D", "0.29", deprecation_message)


class KCrossAttnUpBlock2D(KCrossAttnUpBlock2D):
    deprecation_message = "Importing `KCrossAttnUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KCrossAttnUpBlock2D`, instead."
    deprecate("KCrossAttnUpBlock2D", "0.29", deprecation_message)


# can potentially later be renamed to `No-feed-forward` attention
class KAttentionBlock(KAttentionBlock):
    deprecation_message = "Importing `KAttentionBlock` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KAttentionBlock`, instead."
    deprecate("KAttentionBlock", "0.29", deprecation_message)