# coding=utf-8
# Copyright 2020 Google and The HuggingFace Inc. team.
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
import os
from shutil import copyfile
from typing import Dict, List, Optional, Tuple

import sentencepiece as spm

from ...file_utils import add_start_docstrings
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING, BatchEncoding
from ...utils import logging


SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"google/pegasus-xsum": "https://cdn.huggingface.co/google/pegasus-xsum/spiece.model"}
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/pegasus-xsum": 512,
}


logger = logging.get_logger(__name__)


class PegasusTokenizer(PreTrainedTokenizer):
    r"""
    Construct a PEGASUS tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask_2>"`):
            The token used for masking single token values. This is the token used when training this model with masked
            language modeling (MLM). This is the token that the PEGASUS encoder will try to predict during pretraining.
            It corresponds to `[MASK2]` in `PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive
            Summarization <https://arxiv.org/pdf/1912.08777.pdf>`__.
        mask_token_sent (:obj:`str`, `optional`, defaults to :obj:`"<mask_1>"`):
            The token used for masking whole target sentences. This is the token used when training this model with gap
            sentences generation (GSG). This is the sentence that the PEGASUS decoder will try to predict during
            pretraining. It corresponds to `[MASK1]` in `PEGASUS: Pre-training with Extracted Gap-sentences for
            Abstractive Summarization <https://arxiv.org/pdf/1912.08777.pdf>`__.
        additional_special_tokens (:obj:`List[str]`, `optional`):
            Additional special tokens used by the tokenizer. If no additional_special_tokens are provided <mask_2> and
            <unk_2, ..., unk_102> are used as additional special tokens corresponding to the `original PEGASUS
            tokenizer
            <https://github.com/google-research/pegasus/blob/939830367bcf411193d2b5eca2f2f90f3f9260ca/pegasus/ops/pretrain_parsing_ops.cc#L66>`__
            that uses the tokens 2 - 104 only for pretraining
    """
    vocab_files_names = VOCAB_FILES_NAMES

    offset = 103  # entries 2 - 104 are only used for pretraining
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["attention_mask"]


    @property












    @add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
