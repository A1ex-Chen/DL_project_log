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
""" TF 2.0 XLNet model.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import numpy as np
import tensorflow as tf

from .configuration_xlnet import XLNetConfig
from .modeling_tf_utils import TFPreTrainedModel, TFSharedEmbeddings, TFSequenceSummary, shape_list, get_initializer
from .file_utils import add_start_docstrings


logger = logging.getLogger(__name__)

TF_XLNET_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'xlnet-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-tf_model.h5",
    'xlnet-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-tf_model.h5",
}






ACT2FN = {"gelu": tf.keras.layers.Activation(gelu),
          "relu": tf.keras.activations.relu,
          "swish": tf.keras.layers.Activation(swish)}


class TFXLNetRelativeAttention(tf.keras.layers.Layer):



    @staticmethod




class TFXLNetFeedForward(tf.keras.layers.Layer):


class TFXLNetLayer(tf.keras.layers.Layer):



class TFXLNetLMHead(tf.keras.layers.Layer):




class TFXLNetMainLayer(tf.keras.layers.Layer):







    @staticmethod




class TFXLNetPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = XLNetConfig
    pretrained_model_archive_map = TF_XLNET_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "transformer"


XLNET_START_DOCSTRING = r"""    The XLNet model was proposed in
    `XLNet: Generalized Autoregressive Pretraining for Language Understanding`_
    by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
    XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method
    to learn bidirectional contexts by maximizing the expected likelihood over all permutations
    of the input sequence factorization order.

    The specific attention pattern can be controlled at training and test time using the `perm_mask` input.

    Do to the difficulty of training a fully auto-regressive model over various factorization order,
    XLNet is pretrained using only a sub-set of the output tokens as target which are selected
    with the `target_mapping` input.

    To use XLNet for sequential decoding (i.e. not in fully bi-directional setting), use the `perm_mask` and
    `target_mapping` inputs to control the attention span and outputs (see examples in `examples/run_generation.py`)

    This model is a tf.keras.Model `tf.keras.Model`_ sub-class. Use it as a regular TF 2.0 Keras Model and
    refer to the TF 2.0 documentation for all matter related to general usage and behavior.

    .. _`XLNet: Generalized Autoregressive Pretraining for Language Understanding`:
        http://arxiv.org/abs/1906.08237

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
        config (:class:`~transformers.XLNetConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

XLNET_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            XLNet is a model with relative position embeddings so you can either pad the inputs on
            the right or on the left.
            Indices can be obtained using :class:`transformers.XLNetTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **mems**: (`optional`)
            list of ``Numpy array`` or ``tf.Tensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as output by the model
            (see `mems` output below). Can be used to speed up sequential decoding and attend to longer context.
            To activate mems you need to set up config.mem_len to a positive value which will be the max number of tokens in
            the memory output by the model. E.g. `model = XLNetModel.from_pretrained('xlnet-base-case, mem_len=1024)` will
            instantiate a model which can use up to 1024 tokens of memory (in addition to the input it self).
        **perm_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length, sequence_length)``:
            Mask to indicate the attention pattern for each input token with values selected in ``[0, 1]``:
            If ``perm_mask[k, i, j] = 0``, i attend to j in batch k;
            if ``perm_mask[k, i, j] = 1``, i does not attend to j in batch k.
            If None, each token attends to all the others (full bidirectional attention).
            Only used during pretraining (to define factorization order) or for sequential decoding (generation).
        **target_mapping**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, num_predict, sequence_length)``:
            Mask to indicate the output tokens to use.
            If ``target_mapping[k, i, j] = 1``, the i-th predict in batch k is on the j-th token.
            Only used during pretraining for partial prediction or for sequential decoding (generation).
        **token_type_ids**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The type indices in XLNet are NOT selected in the vocabulary, they can be arbitrary numbers and
            the important thing is that they should be different for tokens which belong to different segments.
            The model will compute relative segment differences from the given type indices:
            0 if the segment id of two tokens are the same, 1 if not.
        **input_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Negative of `attention_mask`, i.e. with 0 for real tokens and 1 for padding.
            Kept for compatibility with the original code base.
            You can only uses one of `input_mask` and `attention_mask`
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are MASKED, ``0`` for tokens that are NOT MASKED.
        **head_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""

@add_start_docstrings("The bare XLNet Model transformer outputing raw hidden-states without any specific head on top.",
                      XLNET_START_DOCSTRING, XLNET_INPUTS_DOCSTRING)
class TFXLNetModel(TFXLNetPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``tf.Tensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **mems**: (`optional`, returned when ``config.mem_len > 0``)
            list of ``tf.Tensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            if config.mem_len > 0 else tuple of None. Can be used to speed up sequential decoding and attend to longer context.
            See details in the docstring of the `mems` input above.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import XLNetTokenizer, TFXLNetModel

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        model = TFXLNetModel.from_pretrained('xlnet-large-cased')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """



@add_start_docstrings("""XLNet Model with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
    XLNET_START_DOCSTRING, XLNET_INPUTS_DOCSTRING)
class TFXLNetLMHeadModel(TFXLNetPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **prediction_scores**: ``tf.Tensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **mems**: (`optional`, returned when ``config.mem_len > 0``)
            list of ``tf.Tensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            if config.mem_len > 0 else tuple of None. Can be used to speed up sequential decoding and attend to longer context.
            See details in the docstring of the `mems` input above.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import XLNetTokenizer, TFXLNetLMHeadModel

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        model = TFXLNetLMHeadModel.from_pretrained('xlnet-large-cased')

        # We show how to setup inputs to predict a next token using a bi-directional context.
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is very <mask>"))[None, :]  # We will predict the masked token
        perm_mask = tf.zeros((1, input_ids.shape[1], input_ids.shape[1]))
        perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
        target_mapping = tf.zeros((1, 1, input_ids.shape[1]))  # Shape [1, 1, seq_length] => let's predict one token
        target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)
        outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)

        next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

    """




@add_start_docstrings("""XLNet Model with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    XLNET_START_DOCSTRING, XLNET_INPUTS_DOCSTRING)
class TFXLNetForSequenceClassification(TFXLNetPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **logits**: ``tf.Tensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **mems**: (`optional`, returned when ``config.mem_len > 0``)
            list of ``tf.Tensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            if config.mem_len > 0 else tuple of None. Can be used to speed up sequential decoding and attend to longer context.
            See details in the docstring of the `mems` input above.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import XLNetTokenizer, TFXLNetForSequenceClassification

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        model = TFXLNetForSequenceClassification.from_pretrained('xlnet-large-cased')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        logits = outputs[0]

    """



# @add_start_docstrings("""XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
#     the hidden-states output to compute `span start logits` and `span end logits`). """,
#     XLNET_START_DOCSTRING, XLNET_INPUTS_DOCSTRING)
# class TFXLNetForQuestionAnswering(TFXLNetPreTrainedModel):
class TFXLNetForQuestionAnsweringSimple(TFXLNetPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **start_scores**: ``tf.Tensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``tf.Tensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **mems**: (`optional`, returned when ``config.mem_len > 0``)
            list of ``tf.Tensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            if config.mem_len > 0 else tuple of None. Can be used to speed up sequential decoding and attend to longer context.
            See details in the docstring of the `mems` input above.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import XLNetTokenizer, TFXLNetForQuestionAnsweringSimple

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = TFXLNetForQuestionAnsweringSimple.from_pretrained('xlnet-base-cased')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        start_scores, end_scores = outputs[:2]

    """


# @add_start_docstrings("""XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
#     the hidden-states output to compute `span start logits` and `span end logits`). """,
#     XLNET_START_DOCSTRING, XLNET_INPUTS_DOCSTRING)
# class TFXLNetForQuestionAnswering(TFXLNetPreTrainedModel):
#     r"""
#     Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
#         **start_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
#             ``tf.Tensor`` of shape ``(batch_size, config.start_n_top)``
#             Log probabilities for the top config.start_n_top start token possibilities (beam-search).
#         **start_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
#             ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``
#             Indices for the top config.start_n_top start token possibilities (beam-search).
#         **end_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
#             ``tf.Tensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
#             Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
#         **end_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
#             ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
#             Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
#         **cls_logits**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
#             ``tf.Tensor`` of shape ``(batch_size,)``
#             Log probabilities for the ``is_impossible`` label of the answers.
#         **mems**:
#             list of ``tf.Tensor`` (one for each layer):
#             that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
#             if config.mem_len > 0 else tuple of None. Can be used to speed up sequential decoding and attend to longer context.
#             See details in the docstring of the `mems` input above.
#         **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
#             list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
#             of shape ``(batch_size, sequence_length, hidden_size)``:
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         **attentions**: (`optional`, returned when ``config.output_attentions=True``)
#             list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#     Examples::

#         tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
#         model = XLMForQuestionAnswering.from_pretrained('xlnet-large-cased')
#         input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
#         start_positions = tf.constant([1])
#         end_positions = tf.constant([3])
#         outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
#         loss, start_scores, end_scores = outputs[:2]

#     """
#     def __init__(self, config, *inputs, **kwargs):
#         super(TFXLNetForQuestionAnswering, self).__init__(config, *inputs, **kwargs)
#         self.start_n_top = config.start_n_top
#         self.end_n_top = config.end_n_top

#         self.transformer = TFXLNetMainLayer(config, name='transformer')
#         self.start_logits = TFPoolerStartLogits(config, name='start_logits')
#         self.end_logits = TFPoolerEndLogits(config, name='end_logits')
#         self.answer_class = TFPoolerAnswerClass(config, name='answer_class')

#     def call(self, inputs, training=False):
#         transformer_outputs = self.transformer(inputs, training=training)
#         hidden_states = transformer_outputs[0]
#         start_logits = self.start_logits(hidden_states, p_mask=p_mask)

#         outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, let's remove the dimension added by batch splitting
#             for x in (start_positions, end_positions, cls_index, is_impossible):
#                 if x is not None and x.dim() > 1:
#                     x.squeeze_(-1)

#             # during training, compute the end logits based on the ground truth of the start position
#             end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

#             loss_fct = CrossEntropyLoss()
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2

#             if cls_index is not None and is_impossible is not None:
#                 # Predict answerability from the representation of CLS and START
#                 cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
#                 loss_fct_cls = nn.BCEWithLogitsLoss()
#                 cls_loss = loss_fct_cls(cls_logits, is_impossible)

#                 # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
#                 total_loss += cls_loss * 0.5

#             outputs = (total_loss,) + outputs

#         else:
#             # during inference, compute the end logits based on beam search
#             bsz, slen, hsz = hidden_states.size()
#             start_log_probs = F.softmax(start_logits, dim=-1) # shape (bsz, slen)

#             start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1) # shape (bsz, start_n_top)
#             start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz) # shape (bsz, start_n_top, hsz)
#             start_states = torch.gather(hidden_states, -2, start_top_index_exp) # shape (bsz, start_n_top, hsz)
#             start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1) # shape (bsz, slen, start_n_top, hsz)

#             hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states) # shape (bsz, slen, start_n_top, hsz)
#             p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
#             end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
#             end_log_probs = F.softmax(end_logits, dim=1) # shape (bsz, slen, start_n_top)

#             end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=1) # shape (bsz, end_n_top, start_n_top)
#             end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
#             end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

#             start_states = torch.einsum("blh,bl->bh", hidden_states, start_log_probs)  # get the representation of START as weighted sum of hidden states
#             cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)  # Shape (batch size,): one single `cls_logits` for each sample

#             outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs

#         # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
#         # or (if labels are provided) (total_loss,)
#         return outputs