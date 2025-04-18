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
"""Tokenization classes for OpenAI GPT."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import logging
import os
import re
import sys
import unicodedata
from io import open

import sacremoses as sm

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
        'xlm-mlm-en-2048': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-vocab.json",
        'xlm-mlm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-vocab.json",
        'xlm-mlm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-vocab.json",
        'xlm-mlm-enro-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-vocab.json",
        'xlm-mlm-tlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-vocab.json",
        'xlm-mlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-vocab.json",
        'xlm-clm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-enfr-1024-vocab.json",
        'xlm-clm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-ende-1024-vocab.json",
        'xlm-mlm-17-1280': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-vocab.json",
        'xlm-mlm-100-1280': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-100-1280-vocab.json",
    },
    'merges_file':
    {
        'xlm-mlm-en-2048': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-merges.txt",
        'xlm-mlm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-merges.txt",
        'xlm-mlm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-merges.txt",
        'xlm-mlm-enro-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-merges.txt",
        'xlm-mlm-tlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-merges.txt",
        'xlm-mlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-merges.txt",
        'xlm-clm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-merges.txt",
        'xlm-clm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-merges.txt",
        'xlm-mlm-17-1280': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-merges.txt",
        'xlm-mlm-100-1280': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-100-1280-merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'xlm-mlm-en-2048': 512,
    'xlm-mlm-ende-1024': 512,
    'xlm-mlm-enfr-1024': 512,
    'xlm-mlm-enro-1024': 512,
    'xlm-mlm-tlm-xnli15-1024': 512,
    'xlm-mlm-xnli15-1024': 512,
    'xlm-clm-enfr-1024': 512,
    'xlm-clm-ende-1024': 512,
    'xlm-mlm-17-1280': 512,
    'xlm-mlm-100-1280': 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    'xlm-mlm-en-2048': {"do_lowercase_and_remove_accent": True},
    'xlm-mlm-ende-1024': { "do_lowercase_and_remove_accent": True,
                            "id2lang": { "0": "de",
                                        "1": "en"},
                           "lang2id": { "de": 0,
                                        "en": 1 }},
    'xlm-mlm-enfr-1024': { "do_lowercase_and_remove_accent": True,
                           "id2lang": { "0": "en",
                                        "1": "fr"},
                           "lang2id": { "en": 0,
                                        "fr": 1 }},
    'xlm-mlm-enro-1024': { "do_lowercase_and_remove_accent": True,
                           "id2lang": { "0": "en",
                                        "1": "ro"},
                           "lang2id": { "en": 0,
                                        "ro": 1 }},
    'xlm-mlm-tlm-xnli15-1024': { "do_lowercase_and_remove_accent": True,
                                 "id2lang": {   "0": "ar",
                                                "1": "bg",
                                                "2": "de",
                                                "3": "el",
                                                "4": "en",
                                                "5": "es",
                                                "6": "fr",
                                                "7": "hi",
                                                "8": "ru",
                                                "9": "sw",
                                                "10": "th",
                                                "11": "tr",
                                                "12": "ur",
                                                "13": "vi",
                                                "14": "zh"},
                                 "lang2id": {   "ar": 0,
                                                "bg": 1,
                                                "de": 2,
                                                "el": 3,
                                                "en": 4,
                                                "es": 5,
                                                "fr": 6,
                                                "hi": 7,
                                                "ru": 8,
                                                "sw": 9,
                                                "th": 10,
                                                "tr": 11,
                                                "ur": 12,
                                                "vi": 13,
                                                "zh": 14 }},
    'xlm-mlm-xnli15-1024': { "do_lowercase_and_remove_accent": True,
                             "id2lang": {   "0": "ar",
                                                "1": "bg",
                                                "2": "de",
                                                "3": "el",
                                                "4": "en",
                                                "5": "es",
                                                "6": "fr",
                                                "7": "hi",
                                                "8": "ru",
                                                "9": "sw",
                                                "10": "th",
                                                "11": "tr",
                                                "12": "ur",
                                                "13": "vi",
                                                "14": "zh"},
                                 "lang2id": {   "ar": 0,
                                                "bg": 1,
                                                "de": 2,
                                                "el": 3,
                                                "en": 4,
                                                "es": 5,
                                                "fr": 6,
                                                "hi": 7,
                                                "ru": 8,
                                                "sw": 9,
                                                "th": 10,
                                                "tr": 11,
                                                "ur": 12,
                                                "vi": 13,
                                                "zh": 14 }},
    'xlm-clm-enfr-1024': { "do_lowercase_and_remove_accent": True,
                           "id2lang": { "0": "en",
                                        "1": "fr"},
                           "lang2id": { "en": 0,
                                        "fr": 1 }},
    'xlm-clm-ende-1024': { "do_lowercase_and_remove_accent": True,
                           "id2lang": { "0": "de",
                                        "1": "en"},
                           "lang2id": { "de": 0,
                                        "en": 1 }},
    'xlm-mlm-17-1280': {"do_lowercase_and_remove_accent": False,
                        "id2lang": {
                            "0": "ar",
                            "1": "de",
                            "2": "en",
                            "3": "es",
                            "4": "fr",
                            "5": "hi",
                            "6": "it",
                            "7": "ja",
                            "8": "ko",
                            "9": "nl",
                            "10": "pl",
                            "11": "pt",
                            "12": "ru",
                            "13": "sv",
                            "14": "tr",
                            "15": "vi",
                            "16": "zh"
                        },
                        "lang2id": {
                            "ar": 0,
                            "de": 1,
                            "en": 2,
                            "es": 3,
                            "fr": 4,
                            "hi": 5,
                            "it": 6,
                            "ja": 7,
                            "ko": 8,
                            "nl": 9,
                            "pl": 10,
                            "pt": 11,
                            "ru": 12,
                            "sv": 13,
                            "tr": 14,
                            "vi": 15,
                            "zh": 16}},
    'xlm-mlm-100-1280': {"do_lowercase_and_remove_accent": False,
                        "id2lang": {
                            "0": "af",
                            "1": "als",
                            "2": "am",
                            "3": "an",
                            "4": "ang",
                            "5": "ar",
                            "6": "arz",
                            "7": "ast",
                            "8": "az",
                            "9": "bar",
                            "10": "be",
                            "11": "bg",
                            "12": "bn",
                            "13": "br",
                            "14": "bs",
                            "15": "ca",
                            "16": "ceb",
                            "17": "ckb",
                            "18": "cs",
                            "19": "cy",
                            "20": "da",
                            "21": "de",
                            "22": "el",
                            "23": "en",
                            "24": "eo",
                            "25": "es",
                            "26": "et",
                            "27": "eu",
                            "28": "fa",
                            "29": "fi",
                            "30": "fr",
                            "31": "fy",
                            "32": "ga",
                            "33": "gan",
                            "34": "gl",
                            "35": "gu",
                            "36": "he",
                            "37": "hi",
                            "38": "hr",
                            "39": "hu",
                            "40": "hy",
                            "41": "ia",
                            "42": "id",
                            "43": "is",
                            "44": "it",
                            "45": "ja",
                            "46": "jv",
                            "47": "ka",
                            "48": "kk",
                            "49": "kn",
                            "50": "ko",
                            "51": "ku",
                            "52": "la",
                            "53": "lb",
                            "54": "lt",
                            "55": "lv",
                            "56": "mk",
                            "57": "ml",
                            "58": "mn",
                            "59": "mr",
                            "60": "ms",
                            "61": "my",
                            "62": "nds",
                            "63": "ne",
                            "64": "nl",
                            "65": "nn",
                            "66": "no",
                            "67": "oc",
                            "68": "pl",
                            "69": "pt",
                            "70": "ro",
                            "71": "ru",
                            "72": "scn",
                            "73": "sco",
                            "74": "sh",
                            "75": "si",
                            "76": "simple",
                            "77": "sk",
                            "78": "sl",
                            "79": "sq",
                            "80": "sr",
                            "81": "sv",
                            "82": "sw",
                            "83": "ta",
                            "84": "te",
                            "85": "th",
                            "86": "tl",
                            "87": "tr",
                            "88": "tt",
                            "89": "uk",
                            "90": "ur",
                            "91": "uz",
                            "92": "vi",
                            "93": "war",
                            "94": "wuu",
                            "95": "yi",
                            "96": "zh",
                            "97": "zh_classical",
                            "98": "zh_min_nan",
                            "99": "zh_yue"
                        },
                        "lang2id": {
                            "af": 0,
                            "als": 1,
                            "am": 2,
                            "an": 3,
                            "ang": 4,
                            "ar": 5,
                            "arz": 6,
                            "ast": 7,
                            "az": 8,
                            "bar": 9,
                            "be": 10,
                            "bg": 11,
                            "bn": 12,
                            "br": 13,
                            "bs": 14,
                            "ca": 15,
                            "ceb": 16,
                            "ckb": 17,
                            "cs": 18,
                            "cy": 19,
                            "da": 20,
                            "de": 21,
                            "el": 22,
                            "en": 23,
                            "eo": 24,
                            "es": 25,
                            "et": 26,
                            "eu": 27,
                            "fa": 28,
                            "fi": 29,
                            "fr": 30,
                            "fy": 31,
                            "ga": 32,
                            "gan": 33,
                            "gl": 34,
                            "gu": 35,
                            "he": 36,
                            "hi": 37,
                            "hr": 38,
                            "hu": 39,
                            "hy": 40,
                            "ia": 41,
                            "id": 42,
                            "is": 43,
                            "it": 44,
                            "ja": 45,
                            "jv": 46,
                            "ka": 47,
                            "kk": 48,
                            "kn": 49,
                            "ko": 50,
                            "ku": 51,
                            "la": 52,
                            "lb": 53,
                            "lt": 54,
                            "lv": 55,
                            "mk": 56,
                            "ml": 57,
                            "mn": 58,
                            "mr": 59,
                            "ms": 60,
                            "my": 61,
                            "nds": 62,
                            "ne": 63,
                            "nl": 64,
                            "nn": 65,
                            "no": 66,
                            "oc": 67,
                            "pl": 68,
                            "pt": 69,
                            "ro": 70,
                            "ru": 71,
                            "scn": 72,
                            "sco": 73,
                            "sh": 74,
                            "si": 75,
                            "simple": 76,
                            "sk": 77,
                            "sl": 78,
                            "sq": 79,
                            "sr": 80,
                            "sv": 81,
                            "sw": 82,
                            "ta": 83,
                            "te": 84,
                            "th": 85,
                            "tl": 86,
                            "tr": 87,
                            "tt": 88,
                            "uk": 89,
                            "ur": 90,
                            "uz": 91,
                            "vi": 92,
                            "war": 93,
                            "wuu": 94,
                            "yi": 95,
                            "zh": 96,
                            "zh_classical": 97,
                            "zh_min_nan": 98,
                            "zh_yue": 99
                        }},
}











class XLMTokenizer(PreTrainedTokenizer):
    """
    BPE tokenizer for XLM

        - Moses preprocessing & tokenization for most supported languages

        - Language specific tokenization for Chinese (Jieba), Japanese (KyTea) and Thai (PyThaiNLP)

        - (optionally) lower case & normalize all inputs text

        - argument ``special_tokens`` and function ``set_special_tokens``, can be used to add additional symbols \
        (ex: "__classify__") to a vocabulary
        
        - `lang2id` attribute maps the languages supported by the model with their ids if provided (automatically set for pretrained vocabularies)

        - `id2lang` attributes does reverse mapping if provided (automatically set for pretrained vocabularies)

        - `do_lowercase_and_remove_accent` controle lower casing and accent (automatically set for pretrained vocabularies)
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES






    @property








