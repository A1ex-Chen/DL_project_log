# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes."""


import collections
import copy
import os
import unicodedata
from typing import Optional

from ...utils import logging
from ..bert.tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer, load_vocab


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "cl-tohoku/bert-base-japanese": "https://huggingface.co/cl-tohoku/bert-base-japanese/resolve/main/vocab.txt",
        "cl-tohoku/bert-base-japanese-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/vocab.txt",
        "cl-tohoku/bert-base-japanese-char": "https://huggingface.co/cl-tohoku/bert-base-japanese-char/resolve/main/vocab.txt",
        "cl-tohoku/bert-base-japanese-char-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "cl-tohoku/bert-base-japanese": 512,
    "cl-tohoku/bert-base-japanese-whole-word-masking": 512,
    "cl-tohoku/bert-base-japanese-char": 512,
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "cl-tohoku/bert-base-japanese": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "subword_tokenizer_type": "wordpiece",
    },
    "cl-tohoku/bert-base-japanese-whole-word-masking": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "subword_tokenizer_type": "wordpiece",
    },
    "cl-tohoku/bert-base-japanese-char": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "subword_tokenizer_type": "character",
    },
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "subword_tokenizer_type": "character",
    },
}


class BertJapaneseTokenizer(BertTokenizer):
    """BERT tokenizer for Japanese text"""

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


    @property





class MecabTokenizer:
    """Runs basic tokenization with MeCab morphological parser."""




class CharacterTokenizer:
    """Runs Character tokenziation."""

