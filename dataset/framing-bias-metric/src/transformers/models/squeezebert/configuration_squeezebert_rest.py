# coding=utf-8
# Copyright 2020 The SqueezeBert authors and The HuggingFace Inc. team.
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
""" SqueezeBERT model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "squeezebert/squeezebert-uncased": "https://huggingface.co/squeezebert/squeezebert-uncased/resolve/main/config.json",
    "squeezebert/squeezebert-mnli": "https://huggingface.co/squeezebert/squeezebert-mnli/resolve/main/config.json",
    "squeezebert/squeezebert-mnli-headless": "https://huggingface.co/squeezebert/squeezebert-mnli-headless/resolve/main/config.json",
}


class SqueezeBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.SqueezeBertModel`. It is used
    to instantiate a SqueezeBERT model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the SqueezeBERT model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.SqueezeBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):

        pad_token_id (:obj:`int`, `optional`, defaults to 0):
            The ID of the token in the word embedding to use as padding.
        embedding_size (:obj:`int`, `optional`, defaults to 768):
            The dimension of the word embedding vectors.

        q_groups (:obj:`int`, `optional`, defaults to 4):
            The number of groups in Q layer.
        k_groups (:obj:`int`, `optional`, defaults to 4):
            The number of groups in K layer.
        v_groups (:obj:`int`, `optional`, defaults to 4):
            The number of groups in V layer.
        post_attention_groups (:obj:`int`, `optional`, defaults to 1):
            The number of groups in the first feed forward network layer.
        intermediate_groups (:obj:`int`, `optional`, defaults to 4):
            The number of groups in the second feed forward network layer.
        output_groups (:obj:`int`, `optional`, defaults to 4):
            The number of groups in the third feed forward network layer.

    Examples::

        >>> from transformers import SqueezeBertModel, SqueezeBertConfig

        >>> # Initializing a SqueezeBERT configuration
        >>> configuration = SqueezeBertConfig()

        >>> # Initializing a model from the configuration above
        >>> model = SqueezeBertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config

    Attributes: pretrained_config_archive_map (Dict[str, str]): A dictionary containing all the available pre-trained
    checkpoints.
    """
    pretrained_config_archive_map = SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "squeezebert"
