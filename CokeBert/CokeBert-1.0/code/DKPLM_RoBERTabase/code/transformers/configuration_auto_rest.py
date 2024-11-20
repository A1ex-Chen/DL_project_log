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

from .configuration_bert import BertConfig
from .configuration_openai import OpenAIGPTConfig
from .configuration_gpt2 import GPT2Config
from .configuration_transfo_xl import TransfoXLConfig
from .configuration_xlnet import XLNetConfig
from .configuration_xlm import XLMConfig
from .configuration_roberta import RobertaConfig
from .configuration_distilbert import DistilBertConfig
from .configuration_ctrl import CTRLConfig
from .configuration_camembert import CamembertConfig

logger = logging.getLogger(__name__)


class AutoConfig(object):
    r""":class:`~transformers.AutoConfig` is a generic configuration class
        that will be instantiated as one of the configuration classes of the library
        when created with the `AutoConfig.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method take care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The base model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertConfig (DistilBERT model)
            - contains `bert`: BertConfig (Bert model)
            - contains `openai-gpt`: OpenAIGPTConfig (OpenAI GPT model)
            - contains `gpt2`: GPT2Config (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLConfig (Transformer-XL model)
            - contains `xlnet`: XLNetConfig (XLNet model)
            - contains `xlm`: XLMConfig (XLM model)
            - contains `roberta`: RobertaConfig (RoBERTa model)
            - contains `camembert`: CamembertConfig (CamemBERT model)
            - contains `ctrl` : CTRLConfig (CTRL model)
        This class cannot be instantiated using `__init__()` (throw an error).
    """

    @classmethod