# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Salesforce CTRL configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
from io import open

from .configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP = {"ctrl": "https://storage.googleapis.com/sf-ctrl/pytorch/ctrl-config.json"}

class CTRLConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `CTRLModel`.

    Args:
        vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `CTRLModel` or a configuration json file.
        n_positions: Number of positional embeddings.
        n_ctx: Size of the causal mask (usually same as n_positions).
        dff: Size of the inner dimension of the FFN.
        n_embd: Dimensionality of the embeddings and hidden states.
        n_layer: Number of hidden layers in the Transformer encoder.
        n_head: Number of attention heads for each attention layer in
            the Transformer encoder.
        layer_norm_epsilon: epsilon to use in the layer norm layers
        resid_pdrop: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        attn_pdrop: The dropout ratio for the attention
            probabilities.
        embd_pdrop: The dropout ratio for the embeddings.
        initializer_range: The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.
    """
    pretrained_config_archive_map = CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP


    @property

    @property

    @property

    @property