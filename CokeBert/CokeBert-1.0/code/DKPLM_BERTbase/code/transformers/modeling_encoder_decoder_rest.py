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
""" Classes to support Encoder-Decoder architectures """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os

import torch
from torch import nn

from .modeling_auto import AutoModel, AutoModelWithLMHead

logger = logging.getLogger(__name__)


class PreTrainedEncoderDecoder(nn.Module):
    r"""
        :class:`~transformers.PreTrainedEncoderDecoder` is a generic model class that will be
        instantiated as a transformer architecture with one of the base model
        classes of the library as encoder and (optionally) another one as
        decoder when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method.
    """


    @classmethod




class Model2Model(PreTrainedEncoderDecoder):
    r"""
        :class:`~transformers.Model2Model` instantiates a Seq2Seq2 model
        where both of the encoder and decoder are of the same family. If the
        name of or that path to a pretrained model is specified the encoder and
        the decoder will be initialized with the pretrained weight (the
        cross-attention will be intialized randomly if its weights are not
        present).

        It is possible to override this behavior and initialize, say, the decoder randomly
        by creating it beforehand as follows

            config = BertConfig.from_pretrained()
            decoder = BertForMaskedLM(config)
            model = Model2Model.from_pretrained('bert-base-uncased', decoder_model=decoder)
    """



    @classmethod


class Model2LSTM(PreTrainedEncoderDecoder):
    @classmethod