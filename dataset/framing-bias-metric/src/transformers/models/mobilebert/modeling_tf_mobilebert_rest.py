# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" TF 2.0 MobileBERT model. """

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFNextSentencePredictorOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFMultipleChoiceLoss,
    TFNextSentencePredictionLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_mobilebert import MobileBertConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MobileBertConfig"
_TOKENIZER_FOR_DOC = "MobileBertTokenizer"

TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/mobilebert-uncased",
    # See all MobileBERT models at https://huggingface.co/models?filter=mobilebert
]


class TFMobileBertIntermediate(tf.keras.layers.Layer):



class TFLayerNorm(tf.keras.layers.LayerNormalization):


class TFNoNorm(tf.keras.layers.Layer):




NORM2FN = {"layer_norm": TFLayerNorm, "no_norm": TFNoNorm}


class TFMobileBertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""







class TFMobileBertSelfAttention(tf.keras.layers.Layer):




class TFMobileBertSelfOutput(tf.keras.layers.Layer):



class TFMobileBertAttention(tf.keras.layers.Layer):




class TFOutputBottleneck(tf.keras.layers.Layer):



class TFMobileBertOutput(tf.keras.layers.Layer):



class TFBottleneckLayer(tf.keras.layers.Layer):



class TFBottleneck(tf.keras.layers.Layer):



class TFFFNOutput(tf.keras.layers.Layer):



class TFFFNLayer(tf.keras.layers.Layer):



class TFMobileBertLayer(tf.keras.layers.Layer):



class TFMobileBertEncoder(tf.keras.layers.Layer):



class TFMobileBertPooler(tf.keras.layers.Layer):



class TFMobileBertPredictionHeadTransform(tf.keras.layers.Layer):



class TFMobileBertLMPredictionHead(tf.keras.layers.Layer):




class TFMobileBertMLMHead(tf.keras.layers.Layer):



@keras_serializable
class TFMobileBertMainLayer(tf.keras.layers.Layer):
    config_class = MobileBertConfig







class TFMobileBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileBertConfig
    base_model_prefix = "mobilebert"


@dataclass
class TFMobileBertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFMobileBertForPreTraining`.

    Args:
        prediction_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
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

    loss: Optional[tf.Tensor] = None
    prediction_logits: tf.Tensor = None
    seq_relationship_logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


MOBILEBERT_START_DOCSTRING = r"""

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
        config (:class:`~transformers.MobileBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

MOBILEBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.MobileBertTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`tf.Tensor` of shape :obj:`({0}, hidden_size)`, `optional`):
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
    "The bare MobileBert Model transformer outputting raw hidden-states without any specific head on top.",
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertModel(TFMobileBertPreTrainedModel):

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/mobilebert-uncased",
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )


@add_start_docstrings(
    """
    MobileBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `next sentence prediction (classification)` head.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForPreTraining(TFMobileBertPreTrainedModel):


    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMobileBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)


@add_start_docstrings("""MobileBert Model with a `language modeling` head on top. """, MOBILEBERT_START_DOCSTRING)
class TFMobileBertForMaskedLM(TFMobileBertPreTrainedModel, TFMaskedLanguageModelingLoss):

    _keys_to_ignore_on_load_missing = [r"pooler"]



    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/mobilebert-uncased",
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )


class TFMobileBertOnlyNSPHead(tf.keras.layers.Layer):



@add_start_docstrings(
    """MobileBert Model with a `next sentence prediction (classification)` head on top. """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForNextSentencePrediction(TFMobileBertPreTrainedModel, TFNextSentencePredictionLoss):

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)


@add_start_docstrings(
    """
    MobileBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForSequenceClassification(TFMobileBertPreTrainedModel, TFSequenceClassificationLoss):

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/mobilebert-uncased",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )


@add_start_docstrings(
    """
    MobileBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForQuestionAnswering(TFMobileBertPreTrainedModel, TFQuestionAnsweringLoss):

    _keys_to_ignore_on_load_missing = [r"pooler"]


    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/mobilebert-uncased",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )


@add_start_docstrings(
    """
    MobileBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForMultipleChoice(TFMobileBertPreTrainedModel, TFMultipleChoiceLoss):

    @property

    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/mobilebert-uncased",
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )


@add_start_docstrings(
    """
    MobileBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForTokenClassification(TFMobileBertPreTrainedModel, TFTokenClassificationLoss):

    _keys_to_ignore_on_load_missing = [r"pooler"]


    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/mobilebert-uncased",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )