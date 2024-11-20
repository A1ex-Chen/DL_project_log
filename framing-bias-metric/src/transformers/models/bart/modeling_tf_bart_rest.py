# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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
"""TF BART model, ported from the fairseq repo."""

import math
import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import ACT2FN
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPast,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)

# Public API
from ...modeling_tf_utils import (
    DUMMY_INPUTS,
    TFPreTrainedModel,
    TFSharedEmbeddings,
    TFWrappedEmbeddings,
    cast_bool_to_primitive,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_bart import BartConfig


_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"

BART_START_DOCSTRING = r"""

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

    Args:
        config (:class:`~transformers.BartConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.TFPreTrainedModel.from_pretrained` method to load the
            model weights.
"""


BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Provide for translation and summarization training. By default, the model will create this tensor by
            shifting the input_ids right, following the paper.
        decoder_attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`):
            will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.
        encoder_outputs (:obj:`tf.FloatTensor`, `optional`):
            hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
            of shape :obj:`(batch_size, sequence_length, hidden_size)` is a sequence of
        past_key_values (:obj:`Tuple[Dict[str: tf.Tensor]]` of length :obj:`config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`). Set to :obj:`False` during training, :obj:`True` during generation
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.TFModelOutput` instead of a plain tuple.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""
LARGE_NEGATIVE = -1e8


logger = logging.get_logger(__name__)








class TFPretrainedBartModel(TFPreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"

    @property
    def dummy_inputs(self):
        pad_token = 1
        input_ids = tf.cast(tf.constant(DUMMY_INPUTS), tf.int32)
        decoder_input_ids = tf.cast(tf.constant(DUMMY_INPUTS), tf.int32)
        dummy_inputs = {
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": tf.math.not_equal(input_ids, pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs

    def _shift_right(self, input_ids):
        # Should maybe be decoder_start_token_id. Change for torch and TF in one PR
        position_0_id = self.config.eos_token_id
        pad_token_id = self.config.pad_token_id
        shifted_input_ids = tf.cast(input_ids, tf.int32)
        shifted_input_ids = tf.roll(shifted_input_ids, 1, axis=-1)
        start_tokens = tf.fill((shape_list(shifted_input_ids)[0], 1), position_0_id)
        shifted_input_ids = tf.concat([start_tokens, shifted_input_ids[:, 1:]], -1)
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = tf.where(
            shifted_input_ids == -100, tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids
        )

        # "Verify that `labels` has only positive values and -100"
        assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.cast(0, tf.int32))

        # Make sure the assertion op is called by wrapping the result in an identity no-op
        with tf.control_dependencies([assert_gte0]):
            shifted_input_ids = tf.identity(shifted_input_ids)

        return shifted_input_ids


# Helper Functions, mostly for making masks




# Helper Modules

PAST_KV_DEPRECATION_WARNING = (
    "The `past_key_value_states` argument is deprecated and will be removed in a future "
    "version, use `past_key_values` instead."
)


class TFEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

    def call(self, x, encoder_padding_mask, training=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, self_attn_weights = self.self_attn(query=x, key=x, key_padding_mask=encoder_padding_mask)
        assert shape_list(x) == shape_list(
            residual
        ), f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(x)}"
        x = tf.nn.dropout(x, rate=self.dropout if training else 0)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = tf.nn.dropout(x, rate=self.activation_dropout if training else 0)
        x = self.fc2(x)
        x = tf.nn.dropout(x, rate=self.dropout if training else 0)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, self_attn_weights


class TFBartEncoder(tf.keras.layers.Layer):
    # config_class = BartConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`TFEncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens: TFSharedEmbeddings, **kwargs):
        super().__init__(**kwargs)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = TFSinusoidalPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                name="embed_positions",
            )
        else:
            self.embed_positions = TFLearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                self.padding_idx,
                config.extra_pos_embeddings,
                name="embed_positions",
            )
        self.layers = [TFEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        self.layernorm_embedding = (
            tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
            if config.normalize_embedding
            else tf.keras.layers.Layer()
        )
        self.layer_norm = (
            tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
            if config.add_final_layer_norm
            else None
        )
        self.return_dict = config.return_dict

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
        training=False,
    ):
        """
        Args:
            input_ids (Tensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (Tensor): indicating which indices are padding tokens

        Returns:
            namedtuple:

                - **x** (Tensor): the last encoder layer's output of shape `(src_len, batch, embed_dim)`

                - **encoder_states** (List[tf.Tensor]): all intermediate hidden states of shape `(src_len, batch,
                  embed_dim)`. Only populated if *output_hidden_states* is True.
                - **all_attentions** (List[tf.Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.return_dict

        # check attention mask and invert
        if attention_mask is not None:
            assert (
                attention_mask._rank() == 2
            ), f"expected attention_mask._rank() to be a 2D tensor got {attention_mask._rank()}"
            attention_mask = tf.cast(attention_mask, dtype=tf.float32)
            attention_mask = (1.0 - attention_mask) * LARGE_NEGATIVE
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = tf.nn.dropout(x, rate=self.dropout if training else 0)

        # B x T x C -> T x B x C
        x = tf.transpose(x, perm=[1, 0, 2])

        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # encoder layers
        for encoder_layer in self.layers:

            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask)

            if output_attentions:
                all_attentions += (attn,)
        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)
            encoder_states = [tf.transpose(hidden_state, perm=(1, 0, 2)) for hidden_state in encoder_states]
        x = tf.transpose(x, perm=(1, 0, 2))
        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return TFBaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)


class TFDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.encoder_attn = TFAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
            name="encoder_attn",
        )
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

    def call(
        self,
        x,
        encoder_hidden_states: tf.Tensor,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        training=False,
    ) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attn_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding elements are indicated by ``1``.
            need_attn_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:

            Tuple containing, encoded output of shape `(seq_len, batch, embed_dim)`, self_attn_weights, layer_state
        """
        residual = x  # Make a copy of the input tensor to add later.
        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # next line mutates layer state and we need a copy of it
        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,
            attn_mask=causal_mask,
            key_padding_mask=decoder_padding_mask,
        )
        x = tf.nn.dropout(x, rate=self.dropout if training else 0)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Cross-Attention Block
        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = tf.nn.dropout(x, rate=self.dropout if training else 0)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = tf.nn.dropout(x, rate=self.activation_dropout if training else 0)
        x = self.fc2(x)
        x = tf.nn.dropout(x, rate=self.dropout if training else 0)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class TFBartDecoder(tf.keras.layers.Layer):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`TFDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens: output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens, **kwargs):
        super().__init__(**kwargs)
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        if config.static_position_embeddings:
            self.embed_positions = TFSinusoidalPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                name="embed_positions",
            )
        else:
            self.embed_positions = TFLearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                self.padding_idx,
                config.extra_pos_embeddings,
                name="embed_positions",
            )
        self.layers = [TFDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        self.layernorm_embedding = (
            tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
            if config.normalize_embedding
            else tf.keras.layers.Layer()
        )
        self.layer_norm = (
            tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
            if config.add_final_layer_norm
            else None
        )

        self.dropout = config.dropout
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache
        self.do_blenderbot_90_layernorm = config.do_blenderbot_90_layernorm

    def call(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_cached_states=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
        training=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if use_cache:
            assert not training, "Training + use cache are incompatible"
        # check attention mask and invert
        use_cache = cast_bool_to_primitive(use_cache)
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]

        x = self.embed_tokens(input_ids) * self.embed_scale
        if self.do_blenderbot_90_layernorm:
            x = self.layernorm_embedding(x) + positions
        else:
            x = self.layernorm_embedding(x + positions)
        x = tf.nn.dropout(x, rate=self.dropout if training else 0)

        # Convert to Bart output format: (BS, seq_len, model_dim) ->  (seq_len, BS, model_dim)
        x = tf.transpose(x, perm=(1, 0, 2))
        assert len(shape_list(encoder_hidden_states)) == 3, "encoder_hidden_states must be a 3D tensor"
        encoder_hidden_states = tf.transpose(encoder_hidden_states, perm=(1, 0, 2))

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if training and (dropout_probability < self.layerdrop):
                continue

            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if output_attentions:
                all_self_attns += (layer_self_attn,)

        if self.layer_norm is not None:  # same as if config.add_final_layer_norm
            x = self.layer_norm(x)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states += (x,)
            # T x B x C -> B x T x C
            all_hidden_states = tuple(tf.transpose(hs, perm=(1, 0, 2)) for hs in all_hidden_states)
        else:
            all_hidden_states = None
        all_self_attns = list(all_self_attns) if output_attentions else None

        x = tf.transpose(x, perm=(1, 0, 2))
        encoder_hidden_states = tf.transpose(encoder_hidden_states, perm=(1, 0, 2))  # could maybe be avoided.

        next_cache = (encoder_hidden_states, next_decoder_cache) if use_cache else None
        if not return_dict:
            return x, next_cache, all_hidden_states, all_self_attns
        else:
            return TFBaseModelOutputWithPast(
                last_hidden_state=x,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )





# Helper Functions, mostly for making masks


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = tf.math.equal(input_ids, padding_idx)  # bool tensor
    return padding_mask


# Helper Modules

PAST_KV_DEPRECATION_WARNING = (
    "The `past_key_value_states` argument is deprecated and will be removed in a future "
    "version, use `past_key_values` instead."
)


class TFEncoderLayer(tf.keras.layers.Layer):



class TFBartEncoder(tf.keras.layers.Layer):
    # config_class = BartConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`TFEncoderLayer`.

    Args:
        config: BartConfig
    """




class TFDecoderLayer(tf.keras.layers.Layer):



class TFBartDecoder(tf.keras.layers.Layer):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`TFDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens: output embedding
    """




def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = tf.gather(input_buffer_k, new_order, axis=0)
    return attn_cache


class TFAttention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""






class TFLearnedPositionalEmbedding(TFSharedEmbeddings):
    """
    This module learns positional embeddings up to a fixed maximum size. Padding ids are ignored by either offsetting
    based on padding_idx or by setting padding_idx to None and ensuring that the appropriate position ids are passed to
    the forward function.
    """




class TFSinusoidalPositionalEmbedding(tf.keras.layers.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""



    @staticmethod



# Public API


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
@keras_serializable
class TFBartModel(TFPretrainedBartModel):


    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )





@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.",
    BART_START_DOCSTRING,
)
class TFBartForConditionalGeneration(TFPretrainedBartModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]


    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)


    @staticmethod



