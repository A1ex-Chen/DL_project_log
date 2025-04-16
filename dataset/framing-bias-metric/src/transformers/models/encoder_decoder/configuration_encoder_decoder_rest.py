# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import copy

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class EncoderDecoderConfig(PretrainedConfig):
    r"""
    :class:`~transformers.EncoderDecoderConfig` is the configuration class to store the configuration of a
    :class:`~transformers.EncoderDecoderModel`. It is used to instantiate an Encoder Decoder model according to the
    specified arguments, defining the encoder and decoder configs.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        kwargs (`optional`):
            Dictionary of keyword arguments. Notably:

                - **encoder** (:class:`~transformers.PretrainedConfig`, `optional`) -- An instance of a configuration
                  object that defines the encoder config.
                - **decoder** (:class:`~transformers.PretrainedConfig`, `optional`) -- An instance of a configuration
                  object that defines the decoder config.

    Examples::

        >>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

        >>> # Initializing a BERT bert-base-uncased style configuration
        >>> config_encoder = BertConfig()
        >>> config_decoder = BertConfig()

        >>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

        >>> # Initializing a Bert2Bert model from the bert-base-uncased style configurations
        >>> model = EncoderDecoderModel(config=config)

        >>> # Accessing the model configuration
        >>> config_encoder = model.config.encoder
        >>> config_decoder  = model.config.decoder
        >>> # set decoder config to causal lm
        >>> config_decoder.is_decoder = True
        >>> config_decoder.add_cross_attention = True

        >>> # Saving the model, including its configuration
        >>> model.save_pretrained('my-model')

        >>> # loading model and config from pretrained folder
        >>> encoder_decoder_config = EncoderDecoderConfig.from_pretrained('my-model')
        >>> model = EncoderDecoderModel.from_pretrained('my-model', config=encoder_decoder_config)
    """
    model_type = "encoder-decoder"
    is_composition = True


    @classmethod
