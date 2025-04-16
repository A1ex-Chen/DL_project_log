# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI GPT."""


import json
import os
import re
from typing import Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from ..bert.tokenization_bert import BasicTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/vocab.json"},
    "merges_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/merges.txt"},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai-gpt": 512,
}






class OpenAIGPTTokenizer(PreTrainedTokenizer):
    """
    Construct a GPT Tokenizer. Based on Byte-Pair-Encoding with the following peculiarities:

    - lowercases all inputs,
    - uses :obj:`SpaCy` tokenizer and :obj:`ftfy` for pre-BPE tokenization if they are installed, fallback to BERT's
      :obj:`BasicTokenizer` if not.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["attention_mask"]


    @property

    @property






