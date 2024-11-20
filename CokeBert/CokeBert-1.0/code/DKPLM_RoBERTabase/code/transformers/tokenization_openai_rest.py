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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import logging
import os
import re
from io import open

from .tokenization_utils import PreTrainedTokenizer
from .tokenization_bert import BasicTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    'vocab_file': 'vocab.json',
    'merges_file': 'merges.txt',
}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'openai-gpt': "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-vocab.json",
    },
    'merges_file':
    {
        'openai-gpt': "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'openai-gpt': 512,
}



class OpenAIGPTTokenizer(PreTrainedTokenizer):
    """
    BPE tokenizer. Peculiarities:
        - lower case all inputs
        - uses SpaCy tokenizer and ftfy for pre-BPE tokenization if they are installed, fallback to BERT's BasicTokenizer if not.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


    @property





