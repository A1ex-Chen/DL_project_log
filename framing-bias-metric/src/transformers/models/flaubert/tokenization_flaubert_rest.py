# coding=utf-8
# Copyright 2019-present CNRS, Facebook Inc. and the HuggingFace Inc. team.
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
"""Tokenization classes for Flaubert, based on XLM."""


import unicodedata

import six

from ...utils import logging
from ..xlm.tokenization_xlm import XLMTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "flaubert/flaubert_small_cased": "https://huggingface.co/flaubert/flaubert_small_cased/resolve/main/vocab.json",
        "flaubert/flaubert_base_uncased": "https://huggingface.co/flaubert/flaubert_base_uncased/resolve/main/vocab.json",
        "flaubert/flaubert_base_cased": "https://huggingface.co/flaubert/flaubert_base_cased/resolve/main/vocab.json",
        "flaubert/flaubert_large_cased": "https://huggingface.co/flaubert/flaubert_large_cased/resolve/main/vocab.json",
    },
    "merges_file": {
        "flaubert/flaubert_small_cased": "https://huggingface.co/flaubert/flaubert_small_cased/resolve/main/merges.txt",
        "flaubert/flaubert_base_uncased": "https://huggingface.co/flaubert/flaubert_base_uncased/resolve/main/merges.txt",
        "flaubert/flaubert_base_cased": "https://huggingface.co/flaubert/flaubert_base_cased/resolve/main/merges.txt",
        "flaubert/flaubert_large_cased": "https://huggingface.co/flaubert/flaubert_large_cased/resolve/main/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "flaubert/flaubert_small_cased": 512,
    "flaubert/flaubert_base_uncased": 512,
    "flaubert/flaubert_base_cased": 512,
    "flaubert/flaubert_large_cased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "flaubert/flaubert_small_cased": {"do_lowercase": False},
    "flaubert/flaubert_base_uncased": {"do_lowercase": True},
    "flaubert/flaubert_base_cased": {"do_lowercase": False},
    "flaubert/flaubert_large_cased": {"do_lowercase": False},
}



    return six_ensure_text(text, encoding="utf-8", errors="ignore")


class FlaubertTokenizer(XLMTokenizer):
    """
    Construct a Flaubert tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses preprocessing and tokenization.
    - Normalizing all inputs text.
    - The arguments ``special_tokens`` and the function ``set_special_tokens``, can be used to add additional symbols
      (like "__classify__") to a vocabulary.
    - The argument :obj:`do_lowercase` controls lower casing (automatically set for pretrained vocabularies).

    This tokenizer inherits from :class:`~transformers.XLMTokenizer`. Please check the superclass for usage examples
    and documentation regarding arguments.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


