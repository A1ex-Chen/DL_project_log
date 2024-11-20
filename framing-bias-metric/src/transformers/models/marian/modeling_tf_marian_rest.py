# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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
"""TF Marian model, ported from the fairseq repo."""

from ...file_utils import add_start_docstrings, is_tf_available
from ...utils import logging
from ..bart.modeling_tf_bart import BART_START_DOCSTRING, LARGE_NEGATIVE, TFBartForConditionalGeneration
from .configuration_marian import MarianConfig


if is_tf_available():
    import tensorflow as tf


_CONFIG_FOR_DOC = "MarianConfig"

START_DOCSTRING = BART_START_DOCSTRING.replace(
    "inherits from :class:`~transformers.TFPreTrainedModel`",
    "inherits from :class:`~transformers.TFBartForConditionalGeneration`",
).replace("BartConfig", _CONFIG_FOR_DOC)


logger = logging.get_logger(__name__)


@add_start_docstrings("Marian model for machine translation", START_DOCSTRING)
class TFMarianMTModel(TFBartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"model.encoder.embed_positions.weight",
        r"model.decoder.embed_positions.weight",
    ]
    config_class = MarianConfig