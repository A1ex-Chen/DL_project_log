# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Classes to support Encoder-Decoder architectures """


from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_encoder_decoder import EncoderDecoderConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "EncoderDecoderConfig"

ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize a sequence-tsequencece model with any pretrained autoencoding model as the
    encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via
    :meth:`~transformers.AutoModel.from_pretrained` function and the decoder is loaded via
    :meth:`~transformers.AutoModelForCausalLM.from_pretrained` function. Cross-attention layers are automatically added
    to the decoder and should be fine-tuned on a downstream generative task, like summarization.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in `Leveraging Pre-trained Checkpoints for Sequence Generation Tasks
    <https://arxiv.org/abs/1907.12461>`__ by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    After such an Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models
    (see the examples for more information).

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

ENCODER_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.PreTrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Provide for sequence to sequence training to the decoder. Indices can be obtained using
            :class:`~transformers.PretrainedTokenizer`. See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for details.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        encoder_outputs (:obj:`tuple(torch.FloatTensor)`, `optional`):
            This tuple must consist of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,
            sequence_length, hidden_size)`) is a tensor of hidden-states at the output of the last layer of the
            encoder. Used in the cross-attention of the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert :obj:`decoder_input_ids`
            indices into associated vectors than the model's internal embedding lookup matrix.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss for the decoder. Indices should be in ``[-100, 0,
            ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            If set to ``True``, the model will return a :class:`~transformers.file_utils.Seq2SeqLMOutput` instead of a
            plain tuple.
        kwargs: (`optional`) Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:

            - Without a prefix which will be input as ``**encoder_kwargs`` for the encoder forward function.
            - With a `decoder_` prefix which will be input as ``**decoder_kwargs`` for the decoder forward function.
"""


@add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
class EncoderDecoderModel(PreTrainedModel):
    r"""
    :class:`~transformers.EncoderDecoder` is a generic model class that will be instantiated as a transformer
    architecture with one of the base model classes of the library as encoder and another one as decoder when created
    with the :meth`~transformers.AutoModel.from_pretrained` class method for the encoder and
    :meth`~transformers.AutoModelForCausalLM.from_pretrained` class method for the decoder.
    """
    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"







    @classmethod

    @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)

