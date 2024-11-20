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
""" PyTorch SqueezeBert model. """


import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_squeezebert import SqueezeBertConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "SqueezeBertConfig"
_TOKENIZER_FOR_DOC = "SqueezeBertTokenizer"

SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "squeezebert/squeezebert-uncased",
    "squeezebert/squeezebert-mnli",
    "squeezebert/squeezebert-mnli-headless",
]


class SqueezeBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""




class MatMulWrapper(torch.nn.Module):
    """
    Wrapper for torch.matmul(). This makes flop-counting easier to implement. Note that if you directly call
    torch.matmul() in your code, the flop counter will typically ignore the flops of the matmul.
    """




class SqueezeBertLayerNorm(nn.LayerNorm):
    """
    This is a nn.LayerNorm subclass that accepts NCW data layout and performs normalization in the C dimension.

    N = batch C = channels W = sequence length
    """




class ConvDropoutLayerNorm(nn.Module):
    """
    ConvDropoutLayerNorm: Conv, Dropout, LayerNorm
    """




class ConvActivation(nn.Module):
    """
    ConvActivation: Conv, Activation
    """




class SqueezeBertSelfAttention(nn.Module):






class SqueezeBertModule(nn.Module):



class SqueezeBertEncoder(nn.Module):



class SqueezeBertPooler(nn.Module):



class SqueezeBertPredictionHeadTransform(nn.Module):



class SqueezeBertLMPredictionHead(nn.Module):



class SqueezeBertOnlyMLMHead(nn.Module):



class SqueezeBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SqueezeBertConfig
    base_model_prefix = "transformer"
    _keys_to_ignore_on_load_missing = [r"position_ids"]



SQUEEZEBERT_START_DOCSTRING = r"""

    The SqueezeBERT model was proposed in `SqueezeBERT: What can computer vision teach NLP about efficient neural
    networks? <https://arxiv.org/abs/2006.11316>`__ by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, and Kurt W.
    Keutzer

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    For best results finetuning SqueezeBERT on text classification tasks, it is recommended to use the
    `squeezebert/squeezebert-mnli-headless` checkpoint as a starting point.

    Parameters:
        config (:class:`~transformers.SqueezeBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.

    Hierarchy::

        Internal class hierarchy:
            SqueezeBertModel
                SqueezeBertEncoder
                    SqueezeBertModule
                    SqueezeBertSelfAttention
                        ConvActivation
                        ConvDropoutLayerNorm

    Data layouts::

        Input data is in [batch, sequence_length, hidden_size] format.

        Data inside the encoder is in [batch, hidden_size, sequence_length] format. But, if :obj:`output_hidden_states
        == True`, the data from inside the encoder is returned in [batch, sequence_length, hidden_size] format.

        The final output of the encoder is in [batch, sequence_length, hidden_size] format.
"""

SQUEEZEBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.SqueezeBertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
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
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

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
    "The bare SqueezeBERT Model transformer outputting raw hidden-states without any specific head on top.",
    SQUEEZEBERT_START_DOCSTRING,
)
class SqueezeBertModel(SqueezeBertPreTrainedModel):




    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="squeezebert/squeezebert-mnli-headless",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )


@add_start_docstrings("""SqueezeBERT Model with a `language modeling` head on top. """, SQUEEZEBERT_START_DOCSTRING)
class SqueezeBertForMaskedLM(SqueezeBertPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"predictions.decoder.bias"]



    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="squeezebert/squeezebert-uncased",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )


@add_start_docstrings(
    """
    SqueezeBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    SQUEEZEBERT_START_DOCSTRING,
)
class SqueezeBertForSequenceClassification(SqueezeBertPreTrainedModel):

    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="squeezebert/squeezebert-mnli-headless",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )


@add_start_docstrings(
    """
    SqueezeBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    SQUEEZEBERT_START_DOCSTRING,
)
class SqueezeBertForMultipleChoice(SqueezeBertPreTrainedModel):

    @add_start_docstrings_to_model_forward(
        SQUEEZEBERT_INPUTS_DOCSTRING.format("(batch_size, num_choices, sequence_length)")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="squeezebert/squeezebert-mnli-headless",
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )


@add_start_docstrings(
    """
    SqueezeBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    SQUEEZEBERT_START_DOCSTRING,
)
class SqueezeBertForTokenClassification(SqueezeBertPreTrainedModel):

    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="squeezebert/squeezebert-mnli-headless",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )


@add_start_docstrings(
    """
     SqueezeBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
     linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
     """,
    SQUEEZEBERT_START_DOCSTRING,
)
class SqueezeBertForQuestionAnswering(SqueezeBertPreTrainedModel):

    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="squeezebert/squeezebert-mnli-headless",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )