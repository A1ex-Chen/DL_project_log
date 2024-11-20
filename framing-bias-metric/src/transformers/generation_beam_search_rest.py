# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team
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

from abc import ABC, abstractmethod
from collections import UserDict
from typing import Optional, Tuple

import torch

from .file_utils import add_start_docstrings


PROCESS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * num_beams, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using any class inheriting from :class:`~transformers.PretrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        next_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2 * num_beams)`):
            Current scores of the top :obj:`2 * num_beams` non-finished beam hypotheses.
        next_tokens (:obj:`torch.LongTensor` of shape :obj:`(batch_size, 2 * num_beams)`):
            :obj:`input_ids` of the tokens corresponding to the top :obj:`2 * num_beams` non-finished beam hypotheses.
        next_indices (:obj:`torch.LongTensor` of shape :obj:`(batch_size, 2 * num_beams)`):
            Beam indices indicating to which beam hypothesis the :obj:`next_tokens` correspond.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        eos_token_id (:obj:`int`, `optional`):
            The id of the `end-of-sequence` token.

    Return:
        :obj:`UserDict`: A dictionary composed of the fields as defined above:

            - **next_beam_scores** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`) -- Updated
              scores of all non-finished beams.
            - **next_beam_tokens** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`) -- Next tokens
              to be added to the non-finished beam_hypotheses.
            - **next_beam_indices** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`) -- Beam indices
              indicating to which beam the next tokens shall be added.

"""

FINALIZE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * num_beams, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using any class inheriting from :class:`~transformers.PretrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        final_beam_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`):
            The final scores of all non-finished beams.
        final_beam_tokens (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`):
            The last tokens to be added to the non-finished beam_hypotheses.
        final_beam_indices (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`):
            The beam indices indicating to which beam the :obj:`final_beam_tokens` shall be added.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        eos_token_id (:obj:`int`, `optional`):
            The id of the `end-of-sequence` token.

    Return:
        :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
        sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
        batches finished early due to the :obj:`eos_token_id`.

"""


class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for :meth:`~transformers.PretrainedModel.beam_search` and
    :meth:`~transformers.PretrainedModel.beam_sample`.
    """

    @abstractmethod
    @add_start_docstrings(PROCESS_INPUTS_DOCSTRING)

    @abstractmethod
    @add_start_docstrings(FINALIZE_INPUTS_DOCSTRING)


class BeamSearchScorer(BeamScorer):
    r"""
    :class:`transformers.BeamScorer` implementing standard beam search decoding.

    Adapted in part from `Facebook's XLM beam search code
    <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

    Args:
        batch_size (:obj:`int`):
            Batch Size of :obj:`input_ids` for which beam search decoding is run in parallel.
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        num_beams (:obj:`int`):
            Number of beams for beam search.
        device (:obj:`torch.device`):
            Defines the device type (*e.g.*, :obj:`"cpu"` or :obj:`"cuda"`) on which this instance of
            :obj:`BeamSearchScorer` will be allocated.
        length_penalty (:obj:`float`, `optional`, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beam_hyps_to_keep (:obj:`int`, `optional`, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            :meth:`~transformer.BeamSearchScorer.finalize`.
    """


    @property




class BeamHypotheses:


