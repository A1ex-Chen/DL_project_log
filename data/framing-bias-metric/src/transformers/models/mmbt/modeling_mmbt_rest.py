# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) HuggingFace Inc. team.
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
"""PyTorch MMBT model. """


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import ModuleUtilsMixin
from ...utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MMBTConfig"


class ModalEmbeddings(nn.Module):
    """Generic Modal Embeddings which takes in an encoder, and a transformer embedding."""




MMBT_START_DOCSTRING = r"""
    MMBT model was proposed in `Supervised Multimodal Bitransformers for Classifying Images and Text
    <https://github.com/facebookresearch/mmbt>`__ by Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Davide Testuggine.
    It's a supervised multimodal bitransformer model that fuses information from text and other image encoders, and
    obtain state-of-the-art performance on various multimodal classification benchmark tasks.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.MMBTConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration.
        transformer (:class: `~nn.Module`): A text transformer that is used by MMBT.
            It should have embeddings, encoder, and pooler attributes.
        encoder (:class: `~nn.Module`): Encoder for the second modality.
            It should take in a batch of modal inputs and return k, n dimension embeddings.
"""

MMBT_INPUTS_DOCSTRING = r"""
    Args:
        input_modal (``torch.FloatTensor`` of shape ``(batch_size, ***)``):
            The other modality data. It will be the shape that the encoder for that type expects. e.g. With an Image
            Encoder, the shape would be (batch_size, channels, height, width)
        input_ids (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``):
            Indices of input sequence tokens in the vocabulary. It does not expect [CLS] token to be added as it's
            appended to the end of other modality embeddings. Indices can be obtained using
            :class:`~transformers.BertTokenizer`. See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        modal_start_tokens (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Optional start token to be added to Other Modality Embedding. [CLS] Most commonly used for classification
            tasks.
        modal_end_tokens (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Optional end token to be added to Other Modality Embedding. [SEP] Most commonly used.
        attention_mask (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        modal_token_type_ids (`optional`) ``torch.LongTensor`` of shape ``(batch_size, modal_sequence_length)``:
            Segment token indices to indicate different portions of the non-text modality. The embeddings from these
            tokens will be summed with the respective token embeddings for the non-text modality.
        position_ids (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        modal_position_ids (``torch.LongTensor`` of shape ``(batch_size, modal_sequence_length)``, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings for the non-text modality.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        encoder_hidden_states (``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

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
    "The bare MMBT Model outputting raw hidden-states without any specific head on top.",
    MMBT_START_DOCSTRING,
)
class MMBTModel(nn.Module, ModuleUtilsMixin):

    @add_start_docstrings_to_model_forward(MMBT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)




@add_start_docstrings(
    """
    MMBT Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    """,
    MMBT_START_DOCSTRING,
    MMBT_INPUTS_DOCSTRING,
)
class MMBTForClassification(nn.Module):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Returns: `Tuple` comprising various elements depending on the configuration (config) and inputs: **loss**:
    (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``: Classification (or
    regression if config.num_labels==1) loss. **logits**: ``torch.FloatTensor`` of shape ``(batch_size,
    config.num_labels)`` Classification (or regression if config.num_labels==1) scores (before SoftMax).
    **hidden_states**: (`optional`, returned when ``output_hidden_states=True``) list of ``torch.FloatTensor`` (one for
    the output of each layer + the output of the embeddings) of shape ``(batch_size, sequence_length, hidden_size)``:
    Hidden-states of the model at the output of each layer plus the initial embedding outputs. **attentions**:
    (`optional`, returned when ``output_attentions=True``) list of ``torch.FloatTensor`` (one for each layer) of shape
    ``(batch_size, num_heads, sequence_length, sequence_length)``: Attentions weights after the attention softmax, used
    to compute the weighted average in the self-attention heads.

    Examples::

        # For example purposes. Not runnable.
        transformer = BertModel.from_pretrained('bert-base-uncased')
        encoder = ImageEncoder(args)
        model = MMBTForClassification(config, transformer, encoder)
        outputs = model(input_modal, input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

