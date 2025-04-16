# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Auto Model class. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from .tokenization_bert import BertTokenizer
from .tokenization_openai import OpenAIGPTTokenizer
from .tokenization_gpt2 import GPT2Tokenizer
from .tokenization_ctrl import CTRLTokenizer
from .tokenization_transfo_xl import TransfoXLTokenizer
from .tokenization_xlnet import XLNetTokenizer
from .tokenization_xlm import XLMTokenizer
from .tokenization_roberta import RobertaTokenizer
from .tokenization_distilbert import DistilBertTokenizer
#from .tokenization_camembert import CamembertTokenizer

logger = logging.getLogger(__name__)

class AutoTokenizer(object):
    r""":class:`~transformers.AutoTokenizer` is a generic tokenizer class
        that will be instantiated as one of the tokenizer classes of the library
        when created with the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method take care of returning the correct tokenizer class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The tokenizer class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `camembert`: CamembertTokenizer (CamemBERT model)
            - contains `distilbert`: DistilBertTokenizer (DistilBert model)
            - contains `roberta`: RobertaTokenizer (RoBERTa model)
            - contains `bert`: BertTokenizer (Bert model)
            - contains `openai-gpt`: OpenAIGPTTokenizer (OpenAI GPT model)
            - contains `gpt2`: GPT2Tokenizer (OpenAI GPT-2 model)
            - contains `ctrl`: CTRLTokenizer (Salesforce CTRL model)
            - contains `transfo-xl`: TransfoXLTokenizer (Transformer-XL model)
            - contains `xlnet`: XLNetTokenizer (XLNet model)
            - contains `xlm`: XLMTokenizer (XLM model)

        This class cannot be instantiated using `__init__()` (throw an error).
    """

    @classmethod