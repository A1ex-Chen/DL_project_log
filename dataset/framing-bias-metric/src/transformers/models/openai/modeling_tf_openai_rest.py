# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
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
""" TF 2.0 OpenAI GPT model."""

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFConv1D,
    TFPreTrainedModel,
    TFSequenceSummary,
    TFSharedEmbeddings,
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_openai import OpenAIGPTConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "OpenAIGPTConfig"
_TOKENIZER_FOR_DOC = "OpenAIGPTTokenizer"

TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai-gpt",
    # See all OpenAI GPT models at https://huggingface.co/models?filter=openai-gpt
]


class TFAttention(tf.keras.layers.Layer):


    @staticmethod






class TFMLP(tf.keras.layers.Layer):



class TFBlock(tf.keras.layers.Layer):



@keras_serializable
class TFOpenAIGPTMainLayer(tf.keras.layers.Layer):
    config_class = OpenAIGPTConfig







class TFOpenAIGPTPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OpenAIGPTConfig
    base_model_prefix = "transformer"


@dataclass
class TFOpenAIGPTDoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: tf.Tensor = None
    mc_logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


OPENAI_GPT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having all
        the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in
        the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`


    Parameters:
        config (:class:`~transformers.OpenAIGPTConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

OPENAI_GPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.OpenAIGPTTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare OpenAI GPT transformer model outputting raw hidden-states without any specific head on top.",
    OPENAI_GPT_START_DOCSTRING,
)
class TFOpenAIGPTModel(TFOpenAIGPTPreTrainedModel):

    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="openai-gpt",
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )


@add_start_docstrings(
    """
    OpenAI GPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    OPENAI_GPT_START_DOCSTRING,
)
class TFOpenAIGPTLMHeadModel(TFOpenAIGPTPreTrainedModel, TFCausalLanguageModelingLoss):


    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="openai-gpt",
        output_type=TFCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )


@add_start_docstrings(
    """
    OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """,
    OPENAI_GPT_START_DOCSTRING,
)
class TFOpenAIGPTDoubleHeadsModel(TFOpenAIGPTPreTrainedModel):


    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFOpenAIGPTDoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)