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

from .modeling_tf_bert import TFBertModel, TFBertForMaskedLM, TFBertForSequenceClassification, TFBertForQuestionAnswering
from .modeling_tf_openai import TFOpenAIGPTModel, TFOpenAIGPTLMHeadModel
from .modeling_tf_gpt2 import TFGPT2Model, TFGPT2LMHeadModel
from .modeling_tf_transfo_xl import TFTransfoXLModel, TFTransfoXLLMHeadModel
from .modeling_tf_xlnet import TFXLNetModel, TFXLNetLMHeadModel, TFXLNetForSequenceClassification, TFXLNetForQuestionAnsweringSimple
from .modeling_tf_xlm import TFXLMModel, TFXLMWithLMHeadModel, TFXLMForSequenceClassification, TFXLMForQuestionAnsweringSimple
from .modeling_tf_roberta import TFRobertaModel, TFRobertaForMaskedLM, TFRobertaForSequenceClassification
from .modeling_tf_distilbert import TFDistilBertModel, TFDistilBertForQuestionAnswering, TFDistilBertForMaskedLM, TFDistilBertForSequenceClassification
from .modeling_tf_ctrl import TFCTRLModel, TFCTRLLMHeadModel

from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)


class TFAutoModel(object):
    r"""
        :class:`~transformers.TFAutoModel` is a generic model class
        that will be instantiated as one of the base model classes of the library
        when created with the `TFAutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The base model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: TFDistilBertModel (DistilBERT model)
            - contains `roberta`: TFRobertaModel (RoBERTa model)
            - contains `bert`: TFBertModel (Bert model)
            - contains `openai-gpt`: TFOpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: TFGPT2Model (OpenAI GPT-2 model)
            - contains `transfo-xl`: TFTransfoXLModel (Transformer-XL model)
            - contains `xlnet`: TFXLNetModel (XLNet model)
            - contains `xlm`: TFXLMModel (XLM model)
            - contains `ctrl`: TFCTRLModel (CTRL model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    @classmethod


class TFAutoModelWithLMHead(object):
    r"""
        :class:`~transformers.TFAutoModelWithLMHead` is a generic model class
        that will be instantiated as one of the language modeling model classes of the library
        when created with the `TFAutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: TFDistilBertForMaskedLM (DistilBERT model)
            - contains `roberta`: TFRobertaForMaskedLM (RoBERTa model)
            - contains `bert`: TFBertForMaskedLM (Bert model)
            - contains `openai-gpt`: TFOpenAIGPTLMHeadModel (OpenAI GPT model)
            - contains `gpt2`: TFGPT2LMHeadModel (OpenAI GPT-2 model)
            - contains `transfo-xl`: TFTransfoXLLMHeadModel (Transformer-XL model)
            - contains `xlnet`: TFXLNetLMHeadModel (XLNet model)
            - contains `xlm`: TFXLMWithLMHeadModel (XLM model)
            - contains `ctrl`: TFCTRLLMHeadModel (CTRL model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    @classmethod


class TFAutoModelForSequenceClassification(object):
    r"""
        :class:`~transformers.TFAutoModelForSequenceClassification` is a generic model class
        that will be instantiated as one of the sequence classification model classes of the library
        when created with the `TFAutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: TFDistilBertForSequenceClassification (DistilBERT model)
            - contains `roberta`: TFRobertaForSequenceClassification (RoBERTa model)
            - contains `bert`: TFBertForSequenceClassification (Bert model)
            - contains `xlnet`: TFXLNetForSequenceClassification (XLNet model)
            - contains `xlm`: TFXLMForSequenceClassification (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    @classmethod


class TFAutoModelForQuestionAnswering(object):
    r"""
        :class:`~transformers.TFAutoModelForQuestionAnswering` is a generic model class
        that will be instantiated as one of the question answering model classes of the library
        when created with the `TFAutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: TFDistilBertForQuestionAnswering (DistilBERT model)
            - contains `bert`: TFBertForQuestionAnswering (Bert model)
            - contains `xlnet`: TFXLNetForQuestionAnswering (XLNet model)
            - contains `xlm`: TFXLMForQuestionAnswering (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    @classmethod