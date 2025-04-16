# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
""" Tokenization classes for ALBERT model."""


import os
from shutil import copyfile
from typing import List, Optional, Tuple

from ...file_utils import is_sentencepiece_available
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging


if is_sentencepiece_available():
    from .tokenization_albert import AlbertTokenizer
else:
    AlbertTokenizer = None

logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/spiece.model",
        "albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/spiece.model",
        "albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/spiece.model",
        "albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/spiece.model",
        "albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/spiece.model",
        "albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/spiece.model",
        "albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/spiece.model",
        "albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/tokenizer.json",
        "albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/tokenizer.json",
        "albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/tokenizer.json",
        "albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/tokenizer.json",
        "albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/tokenizer.json",
        "albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/tokenizer.json",
        "albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/tokenizer.json",
        "albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "albert-base-v1": 512,
    "albert-large-v1": 512,
    "albert-xlarge-v1": 512,
    "albert-xxlarge-v1": 512,
    "albert-base-v2": 512,
    "albert-large-v2": 512,
    "albert-xlarge-v2": 512,
    "albert-xxlarge-v2": 512,
}

SPIECE_UNDERLINE = "▁"


class AlbertTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" ALBERT tokenizer (backed by HuggingFace's `tokenizers` library). Based on `Unigram
    <https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models>`__. This tokenizer
    inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        remove_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
        keep_accents (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to keep accents when tokenizing.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            .. note:: When building a sequence using special tokens, this is not the token that is used for the
            beginning of sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token
            that is used for the end of sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict. Attributes:
        sp_model (:obj:`SentencePieceProcessor`):
            The `SentencePiece` processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = AlbertTokenizer




