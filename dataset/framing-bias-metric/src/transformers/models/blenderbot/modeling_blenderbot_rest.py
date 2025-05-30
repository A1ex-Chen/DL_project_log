#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the;
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
# LICENSE file in the root directory of this source tree.
""""BlenderbotForConditionalGeneration which inherits from BART"""

import torch

from ...file_utils import add_start_docstrings
from ..bart.modeling_bart import BartForConditionalGeneration
from .configuration_blenderbot import BlenderbotConfig


BLENDER_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

"""

BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST = ["facebook/blenderbot-3B", "facebook/blenderbot-90M"]


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BLENDER_START_DOCSTRING
)
class BlenderbotForConditionalGeneration(BartForConditionalGeneration):
    """
    This class overrides :class:`~transformers.BartForConditionalGeneration`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = BlenderbotConfig
