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

from ..utils import deprecate
from .unets.unet_1d_blocks import (
    AttnDownBlock1D,
    AttnUpBlock1D,
    DownBlock1D,
    DownBlock1DNoSkip,
    DownResnetBlock1D,
    Downsample1d,
    MidResTemporalBlock1D,
    OutConv1DBlock,
    OutValueFunctionBlock,
    ResConvBlock,
    SelfAttention1d,
    UNetMidBlock1D,
    UpBlock1D,
    UpBlock1DNoSkip,
    UpResnetBlock1D,
    Upsample1d,
    ValueFunctionMidBlock1D,
)


class DownResnetBlock1D(DownResnetBlock1D):
    deprecation_message = "Importing `DownResnetBlock1D` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import DownResnetBlock1D`, instead."
    deprecate("DownResnetBlock1D", "0.29", deprecation_message)


class UpResnetBlock1D(UpResnetBlock1D):
    deprecation_message = "Importing `UpResnetBlock1D` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import UpResnetBlock1D`, instead."
    deprecate("UpResnetBlock1D", "0.29", deprecation_message)


class ValueFunctionMidBlock1D(ValueFunctionMidBlock1D):
    deprecation_message = "Importing `ValueFunctionMidBlock1D` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import ValueFunctionMidBlock1D`, instead."
    deprecate("ValueFunctionMidBlock1D", "0.29", deprecation_message)


class OutConv1DBlock(OutConv1DBlock):
    deprecation_message = "Importing `OutConv1DBlock` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import OutConv1DBlock`, instead."
    deprecate("OutConv1DBlock", "0.29", deprecation_message)


class OutValueFunctionBlock(OutValueFunctionBlock):
    deprecation_message = "Importing `OutValueFunctionBlock` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import OutValueFunctionBlock`, instead."
    deprecate("OutValueFunctionBlock", "0.29", deprecation_message)


class Downsample1d(Downsample1d):
    deprecation_message = "Importing `Downsample1d` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import Downsample1d`, instead."
    deprecate("Downsample1d", "0.29", deprecation_message)


class Upsample1d(Upsample1d):
    deprecation_message = "Importing `Upsample1d` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import Upsample1d`, instead."
    deprecate("Upsample1d", "0.29", deprecation_message)


class SelfAttention1d(SelfAttention1d):
    deprecation_message = "Importing `SelfAttention1d` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import SelfAttention1d`, instead."
    deprecate("SelfAttention1d", "0.29", deprecation_message)


class ResConvBlock(ResConvBlock):
    deprecation_message = "Importing `ResConvBlock` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import ResConvBlock`, instead."
    deprecate("ResConvBlock", "0.29", deprecation_message)


class UNetMidBlock1D(UNetMidBlock1D):
    deprecation_message = "Importing `UNetMidBlock1D` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import UNetMidBlock1D`, instead."
    deprecate("UNetMidBlock1D", "0.29", deprecation_message)


class AttnDownBlock1D(AttnDownBlock1D):
    deprecation_message = "Importing `AttnDownBlock1D` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import AttnDownBlock1D`, instead."
    deprecate("AttnDownBlock1D", "0.29", deprecation_message)


class DownBlock1D(DownBlock1D):
    deprecation_message = "Importing `DownBlock1D` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import DownBlock1D`, instead."
    deprecate("DownBlock1D", "0.29", deprecation_message)


class DownBlock1DNoSkip(DownBlock1DNoSkip):
    deprecation_message = "Importing `DownBlock1DNoSkip` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import DownBlock1DNoSkip`, instead."
    deprecate("DownBlock1DNoSkip", "0.29", deprecation_message)


class AttnUpBlock1D(AttnUpBlock1D):
    deprecation_message = "Importing `AttnUpBlock1D` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import AttnUpBlock1D`, instead."
    deprecate("AttnUpBlock1D", "0.29", deprecation_message)


class UpBlock1D(UpBlock1D):
    deprecation_message = "Importing `UpBlock1D` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import UpBlock1D`, instead."
    deprecate("UpBlock1D", "0.29", deprecation_message)


class UpBlock1DNoSkip(UpBlock1DNoSkip):
    deprecation_message = "Importing `UpBlock1DNoSkip` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import UpBlock1DNoSkip`, instead."
    deprecate("UpBlock1DNoSkip", "0.29", deprecation_message)


class MidResTemporalBlock1D(MidResTemporalBlock1D):
    deprecation_message = "Importing `MidResTemporalBlock1D` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import MidResTemporalBlock1D`, instead."
    deprecate("MidResTemporalBlock1D", "0.29", deprecation_message)







