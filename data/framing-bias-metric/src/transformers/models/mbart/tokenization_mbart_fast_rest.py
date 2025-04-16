# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

from typing import List, Optional

from tokenizers import processors

from ...file_utils import add_start_docstrings, is_sentencepiece_available
from ...tokenization_utils import BatchEncoding
from ...tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING
from ...utils import logging
from ..xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast


if is_sentencepiece_available():
    from .tokenization_mbart import MBartTokenizer
else:
    MBartTokenizer = None


logger = logging.get_logger(__name__)

_all_mbart_models = ["facebook/mbart-large-en-ro", "facebook/mbart-large-cc25"]
SPM_URL = "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/sentence.bpe.model"
tokenizer_URL = "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/tokenizer.json"

FAIRSEQ_LANGUAGE_CODES = [
    "ar_AR",
    "cs_CZ",
    "de_DE",
    "en_XX",
    "es_XX",
    "et_EE",
    "fi_FI",
    "fr_XX",
    "gu_IN",
    "hi_IN",
    "it_IT",
    "ja_XX",
    "kk_KZ",
    "ko_KR",
    "lt_LT",
    "lv_LV",
    "my_MM",
    "ne_NP",
    "nl_XX",
    "ro_RO",
    "ru_RU",
    "si_LK",
    "tr_TR",
    "vi_VN",
    "zh_CN",
]


class MBartTokenizerFast(XLMRobertaTokenizerFast):
    """
    Construct a "fast" MBART tokenizer (backed by HuggingFace's `tokenizers` library). Based on `BPE
    <https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models>`__.

    :class:`~transformers.MBartTokenizerFast` is a subclass of :class:`~transformers.XLMRobertaTokenizerFast` and adds
    a new :meth:`~transformers.MBartTokenizerFast.prepare_seq2seq_batch`.

    Refer to superclass :class:`~transformers.XLMRobertaTokenizerFast` for usage examples and documentation concerning
    the initialization parameters and other methods.

    .. warning::
        ``prepare_seq2seq_batch`` should be used to encode inputs. Other tokenizer methods like ``encode`` do not work
        properly.

    The tokenization method is ``<tokens> <eos> <language code>`` for source language documents, and ``<language code>
    <tokens> <eos>``` for target language documents.

    Examples::

        >>> from transformers import MBartTokenizerFast
        >>> tokenizer = MBartTokenizerFast.from_pretrained('facebook/mbart-large-en-ro')
        >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
        >>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
        >>> batch: dict = tokenizer.prepare_seq2seq_batch(
        ...     example_english_phrase, src_lang="en_XX", tgt_lang="ro_RO", tgt_texts=expected_translation_romanian, return_tensors="pt"
        ... )
    """

    vocab_files_names = {"vocab_file": "sentencepiece.bpe.model"}
    max_model_input_sizes = {m: 1024 for m in _all_mbart_models}
    pretrained_vocab_files_map = {"vocab_file": {m: SPM_URL for m in _all_mbart_models}}
    slow_tokenizer_class = MBartTokenizer

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []




    @add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)

