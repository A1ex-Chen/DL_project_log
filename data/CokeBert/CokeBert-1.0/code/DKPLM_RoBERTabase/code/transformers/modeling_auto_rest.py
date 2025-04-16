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

from .modeling_bert import BertModel, BertForMaskedLM, BertForSequenceClassification, BertForQuestionAnswering
from .modeling_openai import OpenAIGPTModel, OpenAIGPTLMHeadModel
from .modeling_gpt2 import GPT2Model, GPT2LMHeadModel
from .modeling_ctrl import CTRLModel, CTRLLMHeadModel
from .modeling_transfo_xl import TransfoXLModel, TransfoXLLMHeadModel
from .modeling_xlnet import XLNetModel, XLNetLMHeadModel, XLNetForSequenceClassification, XLNetForQuestionAnswering
from .modeling_xlm import XLMModel, XLMWithLMHeadModel, XLMForSequenceClassification, XLMForQuestionAnswering
from .modeling_roberta import RobertaModel, RobertaForMaskedLM, RobertaForSequenceClassification
from .modeling_distilbert import DistilBertModel, DistilBertForQuestionAnswering, DistilBertForMaskedLM, DistilBertForSequenceClassification
from .modeling_camembert import CamembertModel, CamembertForMaskedLM, CamembertForSequenceClassification, CamembertForMultipleChoice

from .modeling_utils import PreTrainedModel, SequenceSummary

from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)


class AutoModel(object):
    r"""
        :class:`~transformers.AutoModel` is a generic model class
        that will be instantiated as one of the base model classes of the library
        when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The base model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertModel (DistilBERT model)
            - contains `camembert`: CamembertModel (CamemBERT model)
            - contains `roberta`: RobertaModel (RoBERTa model)
            - contains `bert`: BertModel (Bert model)
            - contains `openai-gpt`: OpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: GPT2Model (OpenAI GPT-2 model)
            - contains `ctrl`: CTRLModel (Salesforce CTRL  model)
            - contains `transfo-xl`: TransfoXLModel (Transformer-XL model)
            - contains `xlnet`: XLNetModel (XLNet model)
            - contains `xlm`: XLMModel (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    @classmethod


class AutoModelWithLMHead(object):
    r"""
        :class:`~transformers.AutoModelWithLMHead` is a generic model class
        that will be instantiated as one of the language modeling model classes of the library
        when created with the `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForMaskedLM (DistilBERT model)
            - contains `camembert`: CamembertForMaskedLM (CamemBERT model)
            - contains `roberta`: RobertaForMaskedLM (RoBERTa model)
            - contains `bert`: BertForMaskedLM (Bert model)
            - contains `openai-gpt`: OpenAIGPTLMHeadModel (OpenAI GPT model)
            - contains `gpt2`: GPT2LMHeadModel (OpenAI GPT-2 model)
            - contains `ctrl`: CTRLLMModel (Salesforce CTRL model)
            - contains `transfo-xl`: TransfoXLLMHeadModel (Transformer-XL model)
            - contains `xlnet`: XLNetLMHeadModel (XLNet model)
            - contains `xlm`: XLMWithLMHeadModel (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    @classmethod


class AutoModelForSequenceClassification(object):
    r"""
        :class:`~transformers.AutoModelForSequenceClassification` is a generic model class
        that will be instantiated as one of the sequence classification model classes of the library
        when created with the `AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForSequenceClassification (DistilBERT model)
            - contains `camembert`: CamembertForSequenceClassification (CamemBERT model)
            - contains `roberta`: RobertaForSequenceClassification (RoBERTa model)
            - contains `bert`: BertForSequenceClassification (Bert model)
            - contains `xlnet`: XLNetForSequenceClassification (XLNet model)
            - contains `xlm`: XLMForSequenceClassification (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    @classmethod


class AutoModelForQuestionAnswering(object):
    r"""
        :class:`~transformers.AutoModelForQuestionAnswering` is a generic model class
        that will be instantiated as one of the question answering model classes of the library
        when created with the `AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForQuestionAnswering (DistilBERT model)
            - contains `bert`: BertForQuestionAnswering (Bert model)
            - contains `xlnet`: XLNetForQuestionAnswering (XLNet model)
            - contains `xlm`: XLMForQuestionAnswering (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    @classmethod