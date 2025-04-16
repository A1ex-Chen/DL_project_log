# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch BERT model specific for generation. """


import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutputWithCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..bert.modeling_bert import BertEncoder
from .configuration_bert_generation import BertGenerationConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BertGenerationConfig"
_TOKENIZER_FOR_DOC = "BertGenerationTokenizer"




class BertGenerationEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""




class BertGenerationPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertGenerationConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]



BERT_GENERATION_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BertGenerationConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

BERT_GENERATION_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertGenerationTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.__call__` and :meth:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
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
    "The bare BertGeneration model transformer outputting raw hidden-states without any specific head on top.",
    BERT_GENERATION_START_DOCSTRING,
)
class BertGenerationEncoder(BertGenerationPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    This model should be used when leveraging Bert or Roberta checkpoints for the
    :class:`~transformers.EncoderDecoderModel` class as described in `Leveraging Pre-trained Checkpoints for Sequence
    Generation Tasks <https://arxiv.org/abs/1907.12461>`__ by Sascha Rothe, Shashi Narayan, and Aliaksei Severyn.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """





    @add_start_docstrings_to_model_forward(BERT_GENERATION_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/bert_for_seq_generation_L-24_bbc_encoder",
        output_type=BaseModelOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )


class BertGenerationOnlyLMHead(nn.Module):



@add_start_docstrings(
    """BertGeneration Model with a `language modeling` head on top for CLM fine-tuning. """,
    BERT_GENERATION_START_DOCSTRING,
)
class BertGenerationDecoder(BertGenerationPreTrainedModel):


    @add_start_docstrings_to_model_forward(BERT_GENERATION_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
