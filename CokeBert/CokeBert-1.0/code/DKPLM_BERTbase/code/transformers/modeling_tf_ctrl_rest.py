# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
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
""" TF 2.0 CTRL model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import sys
from io import open
import numpy as np
import tensorflow as tf

from .configuration_ctrl import CTRLConfig
from .modeling_tf_utils import TFPreTrainedModel, get_initializer, shape_list, TFSharedEmbeddings
from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

TF_CTRL_PRETRAINED_MODEL_ARCHIVE_MAP = {"ctrl": "https://s3.amazonaws.com/models.huggingface.co/bert/ctrl-tf_model.h5"}





class TFMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model_size, num_heads, output_attentions=False, **kwargs):
        super(TFMultiHeadAttention, self).__init__(**kwargs)
        self.output_attentions = output_attentions
        self.num_heads = num_heads
        self.d_model_size = d_model_size

        self.depth = int(d_model_size / self.num_heads)

        self.Wq = tf.keras.layers.Dense(d_model_size, name='Wq')
        self.Wk = tf.keras.layers.Dense(d_model_size, name='Wk')
        self.Wv = tf.keras.layers.Dense(d_model_size, name='Wv')

        self.dense = tf.keras.layers.Dense(d_model_size, name='dense')

    def split_into_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        v, k, q, mask, layer_past, attention_mask, head_mask = inputs
        batch_size = q.shape[0]

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=1)
            k = tf.concat((past_key, k), dim=-2)
            v = tf.concat((past_value, v), dim=-2)
        present = tf.stack((k, v), axis=1)

        output = scaled_dot_product_attention(q, k, v, mask, attention_mask, head_mask)
        scaled_attention = tf.transpose(output[0], perm=[0, 2, 1, 3])
        attn = output[1]
        original_size_attention = tf.reshape(scaled_attention,  (batch_size, -1, self.d_model_size))
        output = self.dense(original_size_attention)

        outputs = (output, present)
        if self.output_attentions:
            outputs = outputs + (attn,)
        return outputs








def point_wise_feed_forward_network(d_model_size, dff, name=""):
    return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu', name="0"), 
            tf.keras.layers.Dense(d_model_size, name="2")
        ], name="ffn")


class TFEncoderLayer(tf.keras.layers.Layer):



class TFCTRLMainLayer(tf.keras.layers.Layer):






class TFCTRLPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = CTRLConfig
    pretrained_model_archive_map = TF_CTRL_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "transformer"


CTRL_START_DOCSTRING = r"""    CTRL model was proposed in 
    `CTRL: A Conditional Transformer Language Model for Controllable Generation`_
    by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher.
    It's a causal (unidirectional) transformer pre-trained using language modeling on a very large
    corpus of ~140 GB of text data with the first token reserved as a control code (such as Links, Books, Wikipedia etc.).

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`CTRL: A Conditional Transformer Language Model for Controllable Generation`:
        https://www.github.com/salesforce/ctrl

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.CTRLConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

CTRL_INPUTS_DOCSTRING = r"""    Inputs:
        **input_ids**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            CTRL is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.
            Indices can be obtained using :class:`transformers.CTRLTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **past**:
            list of ``Numpy array`` or ``tf.Tensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding.
        **attention_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices).
        **position_ids**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""

@add_start_docstrings("The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.",
                                            CTRL_START_DOCSTRING, CTRL_INPUTS_DOCSTRING)
class TFCTRLModel(TFCTRLPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``tf.Tensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **past**:
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import CTRLTokenizer, TFCTRLModel

        tokenizer = CTRLTokenizer.from_pretrained('ctrl')
        model = TFCTRLModel.from_pretrained('ctrl')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """



class TFCTRLLMHead(tf.keras.layers.Layer):




@add_start_docstrings("""The CTRL Model transformer with a language modeling head on top
(linear layer with weights tied to the input embeddings). """, CTRL_START_DOCSTRING, CTRL_INPUTS_DOCSTRING)
class TFCTRLLMHeadModel(TFCTRLPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import CTRLTokenizer, TFCTRLLMHeadModel

        tokenizer = CTRLTokenizer.from_pretrained('ctrl')
        model = TFCTRLLMHeadModel.from_pretrained('ctrl')

        input_ids = torch.tensor(tokenizer.encode("Links Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """

