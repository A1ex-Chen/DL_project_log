# Copyright (c) InternLM. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Tokenization classes for IntermLM."""
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': './tokenizer.model'}

PRETRAINED_VOCAB_FILES_MAP = {}


class InternLMXComposer2Tokenizer(PreTrainedTokenizer):
    """Construct a InternLM tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ['input_ids', 'attention_mask']
    _auto_class = 'AutoTokenizer'


    @property

    @property

    @property

    @property









