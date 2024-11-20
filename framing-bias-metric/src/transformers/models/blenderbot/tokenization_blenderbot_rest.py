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
""""BlenderbotTokenizer and BlenderbotSmallTokenizer"""
import json
import os
from typing import Dict, List, Optional, Tuple

import regex as re

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from ..roberta.tokenization_roberta import RobertaTokenizer


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    # "tokenizer_config_file": "tokenizer_config.json",
}
CKPT_3B = "facebook/blenderbot-3B"


class BlenderbotTokenizer(RobertaTokenizer):
    r"""
    Construct a Blenderbot tokenizer.

    :class:`~transformers.Blenderbot` is nearly identical to :class:`~transformers.RobertaTokenizer` and runs
    end-to-end tokenization: punctuation splitting and wordpiece. The only difference is that it doesnt add BOS token
    to the beginning of sequences.

    Refer to superclass :class:`~transformers.RobertaTokenizer` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "tokenizer_config_file": "tokenizer_config.json",
    }
    pretrained_vocab_files_map = {
        "vocab_file": {CKPT_3B: "https://cdn.huggingface.co/facebook/blenderbot-3B/vocab.json"},
        "merges_file": {CKPT_3B: "https://cdn.huggingface.co/facebook/blenderbot-3B/merges.txt"},
        "tokenizer_config_file": {CKPT_3B: "https://cdn.huggingface.co/facebook/blenderbot-3B/tokenizer_config.json"},
    }
    max_model_input_sizes = {"facebook/blenderbot-3B": 128}

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: List[int] = None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Blenderbot sequence has the following format:

        - single sequence: `` X </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`):
                Will be ignored

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        return token_ids_0 + [self.eos_token_id]




def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    pairs = set(pairs)
    return pairs


class BlenderbotSmallTokenizer(PreTrainedTokenizer):
    """
    Constructs a Blenderbot-90M tokenizer based on BPE (Byte-Pair-Encoding)

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        merges_file (:obj:`str`):
            Path to the merges file.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"__start__"`):
            The beginning of sentence token.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"__end__"`):
            The end of sentence token.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"__unk__"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"__pad__"`):
            The token used for padding, for example when batching sequences of different lengths.
        **kwargs
            Additional keyword arguments passed along to :class:`~transformers.PreTrainedTokenizer`
    """

    vocab_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}
    pretrained_vocab_files_map = {
        "vocab_file": {"facebook/blenderbot-90M": "https://cdn.huggingface.co/facebook/blenderbot-90M/vocab.json"},
        "merges_file": {"facebook/blenderbot-90M": "https://cdn.huggingface.co/facebook/blenderbot-90M/merges.txt"},
    }
    max_model_input_sizes = {"facebook/blenderbot-90M": 512}


    @property






