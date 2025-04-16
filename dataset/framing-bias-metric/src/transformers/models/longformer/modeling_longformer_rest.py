# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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
"""PyTorch Longformer model. """

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from ...activations import ACT2FN, gelu
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging
from .configuration_longformer import LongformerConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LongformerConfig"
_TOKENIZER_FOR_DOC = "LongformerTokenizer"

LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "allenai/longformer-large-4096-finetuned-triviaqa",
    "allenai/longformer-base-4096-extra.pos.embd.only",
    "allenai/longformer-large-4096-extra.pos.embd.only",
    # See all Longformer models at https://huggingface.co/models?filter=longformer
]


@dataclass
class LongformerBaseModelOutput(ModelOutput):
    """
    Base class for Longformer's outputs, with potential hidden states, local and global attentions.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x + attention_window + 1)`, where ``x`` is the number of tokens with global attention
            mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first ``x`` values) and to every token in the attention window (remaining
            ``attention_window + 1`` values). Note that the first ``x`` values refer to tokens with fixed positions in
            the text, but the remaining ``attention_window + 1`` values refer to tokens with relative positions: the
            attention weight of a token to itself is located at index ``x + attention_window / 2`` and the
            ``attention_window / 2`` preceding (succeeding) values are the attention weights to the ``attention_window
            / 2`` preceding (succeeding) tokens. If the attention window contains a token with global attention, the
            attention weight at the corresponding index is set to 0; the value should be accessed from the first ``x``
            attention weights. If a token has global attention, the attention weights to all other tokens in
            :obj:`attentions` is set to 0, the values should be accessed from :obj:`global_attentions`.
        global_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x)`, where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LongformerBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for Longformer's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x + attention_window + 1)`, where ``x`` is the number of tokens with global attention
            mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first ``x`` values) and to every token in the attention window (remaining
            ``attention_window + 1`` values). Note that the first ``x`` values refer to tokens with fixed positions in
            the text, but the remaining ``attention_window + 1`` values refer to tokens with relative positions: the
            attention weight of a token to itself is located at index ``x + attention_window / 2`` and the
            ``attention_window / 2`` preceding (succeeding) values are the attention weights to the ``attention_window
            / 2`` preceding (succeeding) tokens. If the attention window contains a token with global attention, the
            attention weight at the corresponding index is set to 0; the value should be accessed from the first ``x``
            attention weights. If a token has global attention, the attention weights to all other tokens in
            :obj:`attentions` is set to 0, the values should be accessed from :obj:`global_attentions`.
        global_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x)`, where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    last_hidden_state: torch.FloatTensor
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LongformerMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Masked language modeling (MLM) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x + attention_window + 1)`, where ``x`` is the number of tokens with global attention
            mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first ``x`` values) and to every token in the attention window (remaining
            ``attention_window + 1`` values). Note that the first ``x`` values refer to tokens with fixed positions in
            the text, but the remaining ``attention_window + 1`` values refer to tokens with relative positions: the
            attention weight of a token to itself is located at index ``x + attention_window / 2`` and the
            ``attention_window / 2`` preceding (succeeding) values are the attention weights to the ``attention_window
            / 2`` preceding (succeeding) tokens. If the attention window contains a token with global attention, the
            attention weight at the corresponding index is set to 0; the value should be accessed from the first ``x``
            attention weights. If a token has global attention, the attention weights to all other tokens in
            :obj:`attentions` is set to 0, the values should be accessed from :obj:`global_attentions`.
        global_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x)`, where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LongformerQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering Longformer models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x + attention_window + 1)`, where ``x`` is the number of tokens with global attention
            mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first ``x`` values) and to every token in the attention window (remaining
            ``attention_window + 1`` values). Note that the first ``x`` values refer to tokens with fixed positions in
            the text, but the remaining ``attention_window + 1`` values refer to tokens with relative positions: the
            attention weight of a token to itself is located at index ``x + attention_window / 2`` and the
            ``attention_window / 2`` preceding (succeeding) values are the attention weights to the ``attention_window
            / 2`` preceding (succeeding) tokens. If the attention window contains a token with global attention, the
            attention weight at the corresponding index is set to 0; the value should be accessed from the first ``x``
            attention weights. If a token has global attention, the attention weights to all other tokens in
            :obj:`attentions` is set to 0, the values should be accessed from :obj:`global_attentions`.
        global_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x)`, where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LongformerSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x + attention_window + 1)`, where ``x`` is the number of tokens with global attention
            mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first ``x`` values) and to every token in the attention window (remaining
            ``attention_window + 1`` values). Note that the first ``x`` values refer to tokens with fixed positions in
            the text, but the remaining ``attention_window + 1`` values refer to tokens with relative positions: the
            attention weight of a token to itself is located at index ``x + attention_window / 2`` and the
            ``attention_window / 2`` preceding (succeeding) values are the attention weights to the ``attention_window
            / 2`` preceding (succeeding) tokens. If the attention window contains a token with global attention, the
            attention weight at the corresponding index is set to 0; the value should be accessed from the first ``x``
            attention weights. If a token has global attention, the attention weights to all other tokens in
            :obj:`attentions` is set to 0, the values should be accessed from :obj:`global_attentions`.
        global_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x)`, where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LongformerMultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice Longformer models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x + attention_window + 1)`, where ``x`` is the number of tokens with global attention
            mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first ``x`` values) and to every token in the attention window (remaining
            ``attention_window + 1`` values). Note that the first ``x`` values refer to tokens with fixed positions in
            the text, but the remaining ``attention_window + 1`` values refer to tokens with relative positions: the
            attention weight of a token to itself is located at index ``x + attention_window / 2`` and the
            ``attention_window / 2`` preceding (succeeding) values are the attention weights to the ``attention_window
            / 2`` preceding (succeeding) tokens. If the attention window contains a token with global attention, the
            attention weight at the corresponding index is set to 0; the value should be accessed from the first ``x``
            attention weights. If a token has global attention, the attention weights to all other tokens in
            :obj:`attentions` is set to 0, the values should be accessed from :obj:`global_attentions`.
        global_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x)`, where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LongformerTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x + attention_window + 1)`, where ``x`` is the number of tokens with global attention
            mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first ``x`` values) and to every token in the attention window (remaining
            ``attention_window + 1`` values). Note that the first ``x`` values refer to tokens with fixed positions in
            the text, but the remaining ``attention_window + 1`` values refer to tokens with relative positions: the
            attention weight of a token to itself is located at index ``x + attention_window / 2`` and the
            ``attention_window / 2`` preceding (succeeding) values are the attention weights to the ``attention_window
            / 2`` preceding (succeeding) tokens. If the attention window contains a token with global attention, the
            attention weight at the corresponding index is set to 0; the value should be accessed from the first ``x``
            attention weights. If a token has global attention, the attention weights to all other tokens in
            :obj:`attentions` is set to 0, the values should be accessed from :obj:`global_attentions`.
        global_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, x)`, where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None






# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids


class LongformerEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """





class LongformerSelfAttention(nn.Module):


    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod



    @staticmethod





# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class LongformerSelfOutput(nn.Module):



class LongformerAttention(nn.Module):




# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class LongformerIntermediate(nn.Module):



# Copied from transformers.models.bert.modeling_bert.BertOutput
class LongformerOutput(nn.Module):



class LongformerLayer(nn.Module):




class LongformerEncoder(nn.Module):



# Copied from transformers.models.bert.modeling_bert.BertPooler
class LongformerPooler(nn.Module):



# Copied from transformers.models.roberta.modeling_roberta.RobertaLMHead with Roberta->Longformer
class LongformerLMHead(nn.Module):
    """Longformer Head for masked language modeling."""




class LongformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LongformerConfig
    base_model_prefix = "longformer"
    _keys_to_ignore_on_load_missing = [r"position_ids"]



LONGFORMER_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.LongformerConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

LONGFORMER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.LongformerTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        global_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to decide the attention given on each token, local attention or global attention. Tokens with global
            attention attends to all other tokens, and all other tokens attend to them. This is important for
            task-specific finetuning because it makes the model more flexible at representing the task. For example,
            for classification, the <s> token should be given global attention. For QA, all question tokens should also
            have global attention. Please refer to the `Longformer paper <https://arxiv.org/abs/2004.05150>`__ for more
            details. Mask values selected in ``[0, 1]``:

            - 0 for local attention (a sliding window attention),
            - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).

        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
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
"""


@add_start_docstrings(
    "The bare Longformer Model outputting raw hidden-states without any specific head on top.",
    LONGFORMER_START_DOCSTRING,
)
class LongformerModel(LongformerPreTrainedModel):
    """
    This class copied code from :class:`~transformers.RobertaModel` and overwrote standard self-attention with
    longformer self-attention to provide the ability to process long sequences following the self-attention approach
    described in `Longformer: the Long-Document Transformer <https://arxiv.org/abs/2004.05150>`__ by Iz Beltagy,
    Matthew E. Peters, and Arman Cohan. Longformer self-attention combines a local (sliding window) and global
    attention to extend to long documents without the O(n^2) increase in memory and compute.

    The self-attention module :obj:`LongformerSelfAttention` implemented here supports the combination of local and
    global attention but it lacks support for autoregressive attention and dilated attention. Autoregressive and
    dilated attention are more relevant for autoregressive language modeling than finetuning on downstream tasks.
    Future release will add support for autoregressive attention, but the support for dilated attention requires a
    custom CUDA kernel to be memory and compute efficient.

    """







    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LongformerBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)


@add_start_docstrings("""Longformer Model with a `language modeling` head on top. """, LONGFORMER_START_DOCSTRING)
class LongformerForMaskedLM(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]



    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LongformerMaskedLMOutput, config_class=_CONFIG_FOR_DOC)


@add_start_docstrings(
    """
    Longformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForSequenceClassification(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]


    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="allenai/longformer-base-4096",
        output_type=LongformerSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )


class LongformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""




@add_start_docstrings(
    """
    Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD /
    TriviaQA (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForQuestionAnswering(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]


    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LongformerQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)


@add_start_docstrings(
    """
    Longformer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForTokenClassification(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]


    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="allenai/longformer-base-4096",
        output_type=LongformerTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )


@add_start_docstrings(
    """
    Longformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForMultipleChoice(LongformerPreTrainedModel):

    @add_start_docstrings_to_model_forward(
        LONGFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="allenai/longformer-base-4096",
        output_type=LongformerMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    is_index_masked,
                    is_index_global_attn,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    is_index_masked=is_index_masked,
                    is_index_global_attn=is_index_global_attn,
                    is_global_attn=is_global_attn,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len + 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)

                if is_global_attn:
                    # bzs x num_attn_heads x num_global_attn x seq_len => bzs x num_attn_heads x seq_len x num_global_attn
                    all_global_attentions = all_global_attentions + (layer_outputs[2].transpose(2, 3),)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_global_attentions] if v is not None
            )
        return LongformerBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            global_attentions=all_global_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class LongformerPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.roberta.modeling_roberta.RobertaLMHead with Roberta->Longformer
class LongformerLMHead(nn.Module):
    """Longformer Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class LongformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LongformerConfig
    base_model_prefix = "longformer"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


LONGFORMER_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.LongformerConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

LONGFORMER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.LongformerTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        global_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to decide the attention given on each token, local attention or global attention. Tokens with global
            attention attends to all other tokens, and all other tokens attend to them. This is important for
            task-specific finetuning because it makes the model more flexible at representing the task. For example,
            for classification, the <s> token should be given global attention. For QA, all question tokens should also
            have global attention. Please refer to the `Longformer paper <https://arxiv.org/abs/2004.05150>`__ for more
            details. Mask values selected in ``[0, 1]``:

            - 0 for local attention (a sliding window attention),
            - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).

        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
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
"""


@add_start_docstrings(
    "The bare Longformer Model outputting raw hidden-states without any specific head on top.",
    LONGFORMER_START_DOCSTRING,
)
class LongformerModel(LongformerPreTrainedModel):
    """
    This class copied code from :class:`~transformers.RobertaModel` and overwrote standard self-attention with
    longformer self-attention to provide the ability to process long sequences following the self-attention approach
    described in `Longformer: the Long-Document Transformer <https://arxiv.org/abs/2004.05150>`__ by Iz Beltagy,
    Matthew E. Peters, and Arman Cohan. Longformer self-attention combines a local (sliding window) and global
    attention to extend to long documents without the O(n^2) increase in memory and compute.

    The self-attention module :obj:`LongformerSelfAttention` implemented here supports the combination of local and
    global attention but it lacks support for autoregressive attention and dilated attention. Autoregressive and
    dilated attention are more relevant for autoregressive language modeling than finetuning on downstream tasks.
    Future release will add support for autoregressive attention, but the support for dilated attention requires a
    custom CUDA kernel to be memory and compute efficient.

    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.embeddings = LongformerEmbeddings(config)
        self.encoder = LongformerEncoder(config)
        self.pooler = LongformerPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logger.info(
                "Input ids are automatically padded from {} to {} to be a multiple of `config.attention_window`: {}".format(
                    seq_len, seq_len + padding_len, attention_window
                )
            )
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = F.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
            token_type_ids = F.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LongformerBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""

        Returns:

        Examples::

            >>> import torch
            >>> from transformers import LongformerModel, LongformerTokenizer

            >>> model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
            >>> tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

            >>> SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
            >>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

            >>> # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
            >>> attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
            >>> global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to global attention to be deactivated for all tokens
            >>> global_attention_mask[:, [1, 4, 21,]] = 1  # Set global attention to random tokens for the sake of this example
            ...                                     # Usually, set global attention based on the task. For example,
            ...                                     # classification: the <s> token
            ...                                     # QA: question tokens
            ...                                     # LM: potentially on the beginning of sentences and paragraphs
            >>> outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
            >>> sequence_output = outputs.last_hidden_state
            >>> pooled_output = outputs.pooler_output
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)[
            :, 0, 0, :
        ]

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # undo padding
        if padding_len > 0:
            # unpad `sequence_output` because the calling function is expecting a length == input_ids.size(1)
            sequence_output = sequence_output[:, :-padding_len]

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return LongformerBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            global_attentions=encoder_outputs.global_attentions,
        )


@add_start_docstrings("""Longformer Model with a `language modeling` head on top. """, LONGFORMER_START_DOCSTRING)
class LongformerForMaskedLM(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LongformerMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> import torch
            >>> from transformers import LongformerForMaskedLM, LongformerTokenizer

            >>> model = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096')
            >>> tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

            >>> SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
            >>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

            >>> attention_mask = None  # default is local attention everywhere, which is a good choice for MaskedLM
            ...                        # check ``LongformerModel.forward`` for more details how to set `attention_mask`
            >>> outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            >>> loss = outputs.loss
            >>> prediction_logits = output.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LongformerMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Longformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForSequenceClassification(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = LongformerClassificationHead(config)

        self.init_weights()

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="allenai/longformer-base-4096",
        output_type=LongformerSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            logger.info("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LongformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


@add_start_docstrings(
    """
    Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD /
    TriviaQA (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForQuestionAnswering(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LongformerQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.

        Returns:

        Examples::

            >>> from transformers import LongformerTokenizer, LongformerForQuestionAnswering
            >>> import torch

            >>> tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
            >>> model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

            >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
            >>> encoding = tokenizer(question, text, return_tensors="pt")
            >>> input_ids = encoding["input_ids"]

            >>> # default is local attention everywhere
            >>> # the forward method will automatically set global attention on question tokens
            >>> attention_mask = encoding["attention_mask"]

            >>> outputs = model(input_ids, attention_mask=attention_mask)
            >>> start_logits = outputs.start_logits
            >>> end_logits = outputs.end_logits
            >>> all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

            >>> answer_tokens = all_tokens[torch.argmax(start_logits) :torch.argmax(end_logits)+1]
            >>> answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens)) # remove space prepending space token

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            if input_ids is None:
                logger.warning(
                    "It is not possible to automatically generate the `global_attention_mask` because input_ids is None. Please make sure that it is correctly set."
                )
            else:
                # set global attention on question tokens automatically
                global_attention_mask = _compute_global_attention_mask(input_ids, self.config.sep_token_id)

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return LongformerQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )


@add_start_docstrings(
    """
    Longformer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForTokenClassification(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="allenai/longformer-base-4096",
        output_type=LongformerTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Longformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForMultipleChoice(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        LONGFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="allenai/longformer-base-4096",
        output_type=LongformerMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        labels=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # set global attention on question tokens
        if global_attention_mask is None and input_ids is not None:
            logger.info("Initializing global attention on multiple choice...")
            # put global attention on all tokens after `config.sep_token_id`
            global_attention_mask = torch.stack(
                [
                    _compute_global_attention_mask(input_ids[:, i], self.config.sep_token_id, before_sep_token=False)
                    for i in range(num_choices)
                ],
                dim=1,
            )

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_global_attention_mask = (
            global_attention_mask.view(-1, global_attention_mask.size(-1))
            if global_attention_mask is not None
            else None
        )
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.longformer(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            global_attention_mask=flat_global_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )