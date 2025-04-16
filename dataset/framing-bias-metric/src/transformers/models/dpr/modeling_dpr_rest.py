# coding=utf-8
# Copyright 2018 DPR Authors
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
""" PyTorch DPR model for Open Domain Question Answering."""


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..bert.modeling_bert import BertModel
from .configuration_dpr import DPRConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DPRConfig"

DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-ctx_encoder-single-nq-base",
    "facebook/dpr-ctx_encoder-multiset-base",
]
DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-question_encoder-single-nq-base",
    "facebook/dpr-question_encoder-multiset-base",
]
DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dpr-reader-single-nq-base",
    "facebook/dpr-reader-multiset-base",
]


##########
# Outputs
##########


@dataclass
class DPRContextEncoderOutput(ModelOutput):
    """
    Class for outputs of :class:`~transformers.DPRQuestionEncoder`.

    Args:
        pooler_output: (:obj:``torch.FloatTensor`` of shape ``(batch_size, embeddings_size)``):
            The DPR encoder outputs the `pooler_output` that corresponds to the context representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed contexts for nearest neighbors queries with questions embeddings.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    pooler_output: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class DPRQuestionEncoderOutput(ModelOutput):
    """
    Class for outputs of :class:`~transformers.DPRQuestionEncoder`.

    Args:
        pooler_output: (:obj:``torch.FloatTensor`` of shape ``(batch_size, embeddings_size)``):
            The DPR encoder outputs the `pooler_output` that corresponds to the question representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed questions for nearest neighbors queries with context embeddings.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    pooler_output: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class DPRReaderOutput(ModelOutput):
    """
    Class for outputs of :class:`~transformers.DPRQuestionEncoder`.

    Args:
        start_logits: (:obj:``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``):
            Logits of the start index of the span for each passage.
        end_logits: (:obj:``torch.FloatTensor`` of shape ``(n_passages, sequence_length)``):
            Logits of the end index of the span for each passage.
        relevance_logits: (:obj:`torch.FloatTensor`` of shape ``(n_passages, )``):
            Outputs of the QA classifier of the DPRReader that corresponds to the scores of each passage to answer the
            question, compared to all the other passages.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    start_logits: torch.FloatTensor
    end_logits: torch.FloatTensor = None
    relevance_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DPREncoder(PreTrainedModel):

    base_model_prefix = "bert_model"



    @property



class DPRSpanPredictor(PreTrainedModel):

    base_model_prefix = "encoder"





##################
# PreTrainedModel
##################


class DPRPretrainedContextEncoder(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = "ctx_encoder"
    _keys_to_ignore_on_load_missing = [r"position_ids"]



class DPRPretrainedQuestionEncoder(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = "question_encoder"
    _keys_to_ignore_on_load_missing = [r"position_ids"]



class DPRPretrainedReader(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = "span_predictor"
    _keys_to_ignore_on_load_missing = [r"position_ids"]



###############
# Actual Models
###############


DPR_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.DPRConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

DPR_ENCODERS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. To match pretraining, DPR input sequence should be
            formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs (for a pair title+text for example):

            ::

                tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1

            (b) For single sequences (for a question for example):

            ::

                tokens:         [CLS] the dog is hairy . [SEP]
                token_type_ids:   0   0   0   0  0     0   0

            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using :class:`~transformers.DPRTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details. attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`,
            `optional`): Mask to avoid performing attention on padding token indices. Mask values selected in ``[0,
            1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__ token_type_ids (:obj:`torch.LongTensor` of
            shape :obj:`(batch_size, sequence_length)`, `optional`): Segment token indices to indicate first and second
            portions of the inputs. Indices are selected in ``[0, 1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_ inputs_embeds (:obj:`torch.FloatTensor` of
            shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`): Optionally, instead of passing
            :obj:`input_ids` you can choose to directly pass an embedded representation. This is useful if you want
            more control over how to convert :obj:`input_ids` indices into associated vectors than the model's internal
            embedding lookup matrix. output_attentions (:obj:`bool`, `optional`): Whether or not to return the
            attentions tensors of all attention layers. See ``attentions`` under returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`): Whether or not to return the hidden states of all layers.
            See ``hidden_states`` under returned tensors for more detail. return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

DPR_READER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids: (:obj:`Tuple[torch.LongTensor]` of shapes :obj:`(n_passages, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. It has to be a sequence triplet with 1) the question
            and 2) the passages titles and 3) the passages texts To match pretraining, DPR :obj:`input_ids` sequence
            should be formatted with [CLS] and [SEP] with the format:

                ``[CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>``

            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using :class:`~transformers.DPRReaderTokenizer`. See this class documentation for
            more details.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(n_passages, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(n_passages, sequence_length, hidden_size)`, `optional`):
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
    "The bare DPRContextEncoder transformer outputting pooler outputs as context representations.",
    DPR_START_DOCSTRING,
)
class DPRContextEncoder(DPRPretrainedContextEncoder):

    @add_start_docstrings_to_model_forward(DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRContextEncoderOutput, config_class=_CONFIG_FOR_DOC)


@add_start_docstrings(
    "The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.",
    DPR_START_DOCSTRING,
)
class DPRQuestionEncoder(DPRPretrainedQuestionEncoder):

    @add_start_docstrings_to_model_forward(DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRQuestionEncoderOutput, config_class=_CONFIG_FOR_DOC)


@add_start_docstrings(
    "The bare DPRReader transformer outputting span predictions.",
    DPR_START_DOCSTRING,
)
class DPRReader(DPRPretrainedReader):

    @add_start_docstrings_to_model_forward(DPR_READER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRReaderOutput, config_class=_CONFIG_FOR_DOC)