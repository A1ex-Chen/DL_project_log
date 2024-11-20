# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" Transformer XL configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
from io import open

from .configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'transfo-xl-wt103': "https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-config.json",
}

class TransfoXLConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `TransfoXLModel`.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `TransfoXLModel` or a configuration json file.
            cutoffs: cutoffs for the adaptive softmax
            d_model: Dimensionality of the model's hidden states.
            d_embed: Dimensionality of the embeddings
            d_head: Dimensionality of the model's heads.
            div_val: divident value for adapative input and softmax
            pre_lnorm: apply LayerNorm to the input instead of the output
            d_inner: Inner dimension in FF
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            tgt_len: number of tokens to predict
            ext_len: length of the extended context
            mem_len: length of the retained previous heads
            same_length: use the same attn length for all tokens
            proj_share_all_but_first: True to share all but first projs, False not to share.
            attn_type: attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
            clamp_len: use the same pos embeddings after clamp_len
            sample_softmax: number of samples in sampled softmax
            adaptive: use adaptive softmax
            tie_weight: tie the word embedding and softmax weights
            dropout: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            dropatt: The dropout ratio for the attention probabilities.
            untie_r: untie relative position biases
            embd_pdrop: The dropout ratio for the embeddings.
            init: parameter initializer to use
            init_range: parameters initialized by U(-init_range, init_range).
            proj_init_std: parameters initialized by N(0, init_std)
            init_std: parameters initialized by N(0, init_std)
    """
    pretrained_config_archive_map = TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP


    @property

    @property

    @vocab_size.setter

    @property

    @property

    @property