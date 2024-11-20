# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import unittest

from diffusers.models.unets.unet_2d_blocks import *  # noqa F403
from diffusers.utils.testing_utils import torch_device

from .test_unet_blocks_common import UNetBlockTesterMixin


class DownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = DownBlock2D  # noqa F405
    block_type = "down"



class ResnetDownsampleBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = ResnetDownsampleBlock2D  # noqa F405
    block_type = "down"



class AttnDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnDownBlock2D  # noqa F405
    block_type = "down"



class CrossAttnDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = CrossAttnDownBlock2D  # noqa F405
    block_type = "down"




class SimpleCrossAttnDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = SimpleCrossAttnDownBlock2D  # noqa F405
    block_type = "down"

    @property


    @unittest.skipIf(torch_device == "mps", "MPS result is not consistent")


class SkipDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = SkipDownBlock2D  # noqa F405
    block_type = "down"

    @property



class AttnSkipDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnSkipDownBlock2D  # noqa F405
    block_type = "down"

    @property



class DownEncoderBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = DownEncoderBlock2D  # noqa F405
    block_type = "down"

    @property




class AttnDownEncoderBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnDownEncoderBlock2D  # noqa F405
    block_type = "down"

    @property




class UNetMidBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UNetMidBlock2D  # noqa F405
    block_type = "mid"




class UNetMidBlock2DCrossAttnTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UNetMidBlock2DCrossAttn  # noqa F405
    block_type = "mid"




class UNetMidBlock2DSimpleCrossAttnTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UNetMidBlock2DSimpleCrossAttn  # noqa F405
    block_type = "mid"

    @property




class UpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UpBlock2D  # noqa F405
    block_type = "up"

    @property



class ResnetUpsampleBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = ResnetUpsampleBlock2D  # noqa F405
    block_type = "up"

    @property



class CrossAttnUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = CrossAttnUpBlock2D  # noqa F405
    block_type = "up"

    @property




class SimpleCrossAttnUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = SimpleCrossAttnUpBlock2D  # noqa F405
    block_type = "up"

    @property




class AttnUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnUpBlock2D  # noqa F405
    block_type = "up"

    @property

    @unittest.skipIf(torch_device == "mps", "MPS result is not consistent")


class SkipUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = SkipUpBlock2D  # noqa F405
    block_type = "up"

    @property



class AttnSkipUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnSkipUpBlock2D  # noqa F405
    block_type = "up"

    @property



class UpDecoderBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UpDecoderBlock2D  # noqa F405
    block_type = "up"

    @property




class AttnUpDecoderBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnUpDecoderBlock2D  # noqa F405
    block_type = "up"

    @property

