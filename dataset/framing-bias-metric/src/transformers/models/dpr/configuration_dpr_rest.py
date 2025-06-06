# coding=utf-8
# Copyright 2010, DPR authors
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
""" DPR model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

DPR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/dpr-ctx_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/config.json",
    "facebook/dpr-question_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/resolve/main/config.json",
    "facebook/dpr-reader-single-nq-base": "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/config.json",
    "facebook/dpr-ctx_encoder-multiset-base": "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/config.json",
    "facebook/dpr-question_encoder-multiset-base": "https://huggingface.co/facebook/dpr-question_encoder-multiset-base/resolve/main/config.json",
    "facebook/dpr-reader-multiset-base": "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/config.json",
}


class DPRConfig(PretrainedConfig):
    r"""
    :class:`~transformers.DPRConfig` is the configuration class to store the configuration of a `DPRModel`.

    This is the configuration class to store the configuration of a :class:`~transformers.DPRContextEncoder`,
    :class:`~transformers.DPRQuestionEncoder`, or a :class:`~transformers.DPRReader`. It is used to instantiate the
    components of the DPR model.

    This class is a subclass of :class:`~transformers.BertConfig`. Please check the superclass for the documentation of
    all kwargs.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the DPR model. Defines the different tokens that can be represented by the `inputs_ids`
            passed to the forward method of :class:`~transformers.BertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
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
            The vocabulary size of the `token_type_ids` passed into :class:`~transformers.BertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        projection_dim (:obj:`int`, `optional`, defaults to 0):
            Dimension of the projection for the context and question encoders. If it is set to zero (default), then no
            projection is done.
    """
    model_type = "dpr"
