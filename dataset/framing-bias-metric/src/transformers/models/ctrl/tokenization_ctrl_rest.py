# coding=utf-8
# Copyright 2018 Salesforce and The HuggingFace Inc. team.
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
"""Tokenization classes for Salesforce CTRL."""


import json
import os
from typing import Optional, Tuple

import regex as re

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"ctrl": "https://raw.githubusercontent.com/salesforce/ctrl/master/ctrl-vocab.json"},
    "merges_file": {"ctrl": "https://raw.githubusercontent.com/salesforce/ctrl/master/ctrl-merges.txt"},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "ctrl": 256,
}

CONTROL_CODES = {
    "Pregnancy": 168629,
    "Christianity": 7675,
    "Explain": 106423,
    "Fitness": 63440,
    "Saving": 63163,
    "Ask": 27171,
    "Ass": 95985,
    "Joke": 163509,
    "Questions": 45622,
    "Thoughts": 49605,
    "Retail": 52342,
    "Feminism": 164338,
    "Writing": 11992,
    "Atheism": 192263,
    "Netflix": 48616,
    "Computing": 39639,
    "Opinion": 43213,
    "Alone": 44967,
    "Funny": 58917,
    "Gaming": 40358,
    "Human": 4088,
    "India": 1331,
    "Joker": 77138,
    "Diet": 36206,
    "Legal": 11859,
    "Norman": 4939,
    "Tip": 72689,
    "Weight": 52343,
    "Movies": 46273,
    "Running": 23425,
    "Science": 2090,
    "Horror": 37793,
    "Confession": 60572,
    "Finance": 12250,
    "Politics": 16360,
    "Scary": 191985,
    "Support": 12654,
    "Technologies": 32516,
    "Teenage": 66160,
    "Event": 32769,
    "Learned": 67460,
    "Notion": 182770,
    "Wikipedia": 37583,
    "Books": 6665,
    "Extract": 76050,
    "Confessions": 102701,
    "Conspiracy": 75932,
    "Links": 63674,
    "Narcissus": 150425,
    "Relationship": 54766,
    "Relationships": 134796,
    "Reviews": 41671,
    "News": 4256,
    "Translation": 26820,
    "multilingual": 128406,
}




class CTRLTokenizer(PreTrainedTokenizer):
    """
    Construct a CTRL tokenizer. Based on Byte-Pair-Encoding.

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
    control_codes = CONTROL_CODES


    @property








    # def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
    #     filtered_tokens = ' '.join(self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens))
    #     tokens_generated_so_far = re.sub('(@@ )', '', string=filtered_tokens)
    #     tokens_generated_so_far = re.sub('(@@ ?$)', '', string=tokens_generated_so_far)
    #     return ''.join(tokens_generated_so_far)