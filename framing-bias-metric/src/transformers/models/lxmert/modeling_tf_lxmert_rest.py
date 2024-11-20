# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team, and the
# Lxmert Authors.
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
""" TF 2.0 LXMERT model. """

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_utils import TFPreTrainedModel, get_initializer, input_processing, keras_serializable, shape_list
from ...utils import logging
from .configuration_lxmert import LxmertConfig


logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "LxmertConfig"
_TOKENIZER_FOR_DOC = "LxmertTokenizer"

TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "unc-nlp/lxmert-base-uncased",
]


@dataclass
class TFLxmertModelOutput(ModelOutput):
    """
    Lxmert's outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the "relation-ship"
    encoder")


    Args:
        language_output (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the language encoder.
        vision_output (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the visual encoder.
        pooled_output (:obj:`tf.Tensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
            by a Linear layer and a Tanh activation function. The Linear
        language_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    language_output: Optional[tf.Tensor] = None
    vision_output: Optional[tf.Tensor] = None
    pooled_output: Optional[tf.Tensor] = None
    language_hidden_states: Optional[Tuple[tf.Tensor]] = None
    vision_hidden_states: Optional[Tuple[tf.Tensor]] = None
    language_attentions: Optional[Tuple[tf.Tensor]] = None
    vision_attentions: Optional[Tuple[tf.Tensor]] = None
    cross_encoder_attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFLxmertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.LxmertForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``tf.Tensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cross_relationship_score: (:obj:`tf.Tensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the textual matching objective (classification) head (scores of True/False
            continuation before SoftMax).
        question_answering_score: (:obj:`tf.Tensor` of shape :obj:`(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.

    """

    loss: [tf.Tensor] = None
    prediction_logits: Optional[tf.Tensor] = None
    cross_relationship_score: Optional[tf.Tensor] = None
    question_answering_score: Optional[tf.Tensor] = None
    language_hidden_states: Optional[Tuple[tf.Tensor]] = None
    vision_hidden_states: Optional[Tuple[tf.Tensor]] = None
    language_attentions: Optional[Tuple[tf.Tensor]] = None
    vision_attentions: Optional[Tuple[tf.Tensor]] = None
    cross_encoder_attentions: Optional[Tuple[tf.Tensor]] = None


class TFLxmertVisualFeatureEncoder(tf.keras.layers.Layer):



class TFLxmertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""







class TFLxmertAttention(tf.keras.layers.Layer):




class TFLxmertIntermediate(tf.keras.layers.Layer):



class TFLxmertOutput(tf.keras.layers.Layer):



class TFLxmertAttentionOutput(tf.keras.layers.Layer):



class TFLxmertSelfAttentionLayer(tf.keras.layers.Layer):



class TFLxmertCrossAttentionLayer(tf.keras.layers.Layer):



class TFLxmertLayer(tf.keras.layers.Layer):



class TFLxmertXLayer(tf.keras.layers.Layer):






class TFLxmertEncoder(tf.keras.layers.Layer):



@keras_serializable
class TFLxmertMainLayer(tf.keras.layers.Layer):
    config_class = LxmertConfig

    @property








class TFLxmertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LxmertConfig
    base_model_prefix = "lxmert"

    @property


LXMERT_START_DOCSTRING = r"""

    The LXMERT model was proposed in `LXMERT: Learning Cross-Modality Encoder Representations from Transformers
    <https://arxiv.org/abs/1908.07490>`__ by Hao Tan and Mohit Bansal. It's a vision and language transformer model,
    pre-trained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MCSCOCO captions, and Visual genome,
    using a combination of masked language modeling, region of interest feature regression, cross entropy loss for
    question answering attribute prediction, and object tag prediction.

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
        config (:class:`~transformers.LxmertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

LXMERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.LxmertTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        visual_feats: (:obj:`tf.Tensor` of shape :obj:՝(batch_size, num_visual_features, visual_feat_dim)՝):
            This input represents visual features. They ROI pooled object features from bounding boxes using a
            faster-RCNN model)

            These are currently not provided by the transformers library.
        visual_pos: (:obj:`tf.Tensor` of shape :obj:՝(batch_size, num_visual_features, visual_feat_dim)՝):
            This input represents spacial features corresponding to their relative (via index) visual features. The
            pre-trained LXMERT model expects these spacial features to be normalized bounding boxes on a scale of 0 to
            1.

            These are currently not provided by the transformers library.
        attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        visual_attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            MMask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        inputs_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
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
    "The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.",
    LXMERT_START_DOCSTRING,
)
class TFLxmertModel(TFLxmertPreTrainedModel):

    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="unc-nlp/lxmert-base-uncased",
        output_type=TFLxmertModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )


class TFLxmertPooler(tf.keras.layers.Layer):



class TFLxmertPredictionHeadTransform(tf.keras.layers.Layer):



class TFLxmertLMPredictionHead(tf.keras.layers.Layer):




class TFLxmertMLMHead(tf.keras.layers.Layer):



class TFLxmertPreTrainingHeads(tf.keras.layers.Layer):



class TFLxmertVisualAnswerHead(tf.keras.layers.Layer):



class TFLxmertVisualObjHead(tf.keras.layers.Layer):



@add_start_docstrings("""Lxmert Model with a `language modeling` head on top. """, LXMERT_START_DOCSTRING)
class TFLxmertForPreTraining(TFLxmertPreTrainedModel):

    @property

    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFLxmertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)