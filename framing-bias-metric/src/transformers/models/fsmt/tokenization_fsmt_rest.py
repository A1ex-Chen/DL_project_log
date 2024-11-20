# coding=utf-8
# Copyright 2019 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for FSMT."""


import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import sacremoses as sm

from ...file_utils import add_start_docstrings
from ...tokenization_utils import BatchEncoding, PreTrainedTokenizer
from ...tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "src_vocab_file": "vocab-src.json",
    "tgt_vocab_file": "vocab-tgt.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "src_vocab_file": {"stas/tiny-wmt19-en-de": "https://cdn.huggingface.co/stas/tiny-wmt19-en-de/vocab-src.json"},
    "tgt_vocab_file": {"stas/tiny-wmt19-en-de": "https://cdn.huggingface.co/stas/tiny-wmt19-en-de/vocab-tgt.json"},
    "merges_file": {"stas/tiny-wmt19-en-de": "https://cdn.huggingface.co/stas/tiny-wmt19-en-de/merges.txt"},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"stas/tiny-wmt19-en-de": 1024}
PRETRAINED_INIT_CONFIGURATION = {
    "stas/tiny-wmt19-en-de": {
        "langs": ["en", "de"],
        "model_max_length": 1024,
        "special_tokens_map_file": None,
        "full_tokenizer_file": None,
    }
}








# Porting notes:
# this one is modeled after XLMTokenizer
#
# added:
# - src_vocab_file,
# - tgt_vocab_file,
# - langs,


class FSMTTokenizer(PreTrainedTokenizer):
    """
    Construct an FAIRSEQ Transformer tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses preprocessing and tokenization.
    - Normalizing all inputs text.
    - The arguments ``special_tokens`` and the function ``set_special_tokens``, can be used to add additional symbols
      (like "__classify__") to a vocabulary.
    - The argument :obj:`langs` defines a pair of languages.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        langs (:obj:`List[str]`):
            A list of two languages to translate from and to, for instance :obj:`["en", "ru"]`.
        src_vocab_file (:obj:`str`):
            File containing the vocabulary for the source language.
        tgt_vocab_file (:obj:`st`):
            File containing the vocabulary for the target language.
        merges_file (:obj:`str`):
            File containing the merges.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the :obj:`cls_token`.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.

    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


    # hack override

    # hack override
    @property





    @property

    @property











    @add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
