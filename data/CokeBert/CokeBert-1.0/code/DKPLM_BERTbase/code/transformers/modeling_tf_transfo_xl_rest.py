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
""" TF 2.0 Transformer XL model.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import math
import logging
import collections
import sys
from io import open

import numpy as np
import tensorflow as tf

from .configuration_transfo_xl import TransfoXLConfig
from .modeling_tf_utils import TFPreTrainedModel, TFConv1D, TFSequenceSummary, shape_list, get_initializer
from .modeling_tf_transfo_xl_utilities import TFAdaptiveSoftmaxMask
from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'transfo-xl-wt103': "https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-tf_model.h5",
}

class TFPositionalEmbedding(tf.keras.layers.Layer):



class TFPositionwiseFF(tf.keras.layers.Layer):



class TFRelPartialLearnableMultiHeadAttn(tf.keras.layers.Layer):





class TFRelPartialLearnableDecoderLayer(tf.keras.layers.Layer):



class TFAdaptiveEmbedding(tf.keras.layers.Layer):




class TFTransfoXLMainLayer(tf.keras.layers.Layer):











class TFTransfoXLPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = TransfoXLConfig
    pretrained_model_archive_map = TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "transformer"


TRANSFO_XL_START_DOCSTRING = r"""    The Transformer-XL model was proposed in
    `Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context`_
    by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
    It's a causal (uni-directional) transformer with relative positioning (sinusoïdal) embeddings which can reuse
    previously computed hidden-states to attend to longer context (memory).
    This model also uses adaptive softmax inputs and outputs (tied).

    This model is a tf.keras.Model `tf.keras.Model`_ sub-class. Use it as a regular TF 2.0 Keras Model and
    refer to the TF 2.0 documentation for all matter related to general usage and behavior.

    .. _`Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context`:
        https://arxiv.org/abs/1901.02860

    .. _`tf.keras.Model`:
        https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model

    Note on the model inputs:
        TF 2.0 models accepts two formats as inputs:

            - having all inputs as keyword arguments (like PyTorch models), or
            - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is usefull when using `tf.keras.Model.fit()` method which currently requires having all the tensors in the first argument of the model call function: `model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the first positional argument :

        - a single Tensor with input_ids only and nothing else: `model(inputs_ids)
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
            `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associaed to the input names given in the docstring:
            `model({'input_ids': input_ids, 'token_type_ids': token_type_ids})`

    Parameters:
        config (:class:`~transformers.TransfoXLConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

TRANSFO_XL_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            Transformer-XL is a model with relative position embeddings so you can either pad the inputs on
            the right or on the left.
            Indices can be obtained using :class:`transformers.TransfoXLTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **mems**: (`optional`)
            list of ``Numpy array`` or ``tf.Tensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `mems` output below). Can be used to speed up sequential decoding and attend to longer context.
        **head_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""

@add_start_docstrings("The bare Bert Model transformer outputing raw hidden-states without any specific head on top.",
                      TRANSFO_XL_START_DOCSTRING, TRANSFO_XL_INPUTS_DOCSTRING)
class TFTransfoXLModel(TFTransfoXLPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``tf.Tensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **mems**:
            list of ``tf.Tensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `mems` input above). Can be used to speed up sequential decoding and attend to longer context.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import TransfoXLTokenizer, TFTransfoXLModel

        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        model = TFTransfoXLModel.from_pretrained('transfo-xl-wt103')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states, mems = outputs[:2]

    """



@add_start_docstrings("""The Transformer-XL Model with a language modeling head on top
    (adaptive softmax with weights tied to the adaptive input embeddings)""",
    TRANSFO_XL_START_DOCSTRING, TRANSFO_XL_INPUTS_DOCSTRING)
class TFTransfoXLLMHeadModel(TFTransfoXLPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **prediction_scores**: ``None`` if ``lm_labels`` is provided else ``tf.Tensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            We don't output them when the loss is computed to speedup adaptive softmax decoding.
        **mems**:
            list of ``tf.Tensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `mems` input above). Can be used to speed up sequential decoding and attend to longer context.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import TransfoXLTokenizer, TFTransfoXLLMHeadModel

        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        model = TFTransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, mems = outputs[:2]

    """


