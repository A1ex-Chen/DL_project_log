# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional
import sys

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from fairseq.ngram_repeat_block import NGramRepeatBlock

class SequenceGenerator(nn.Module):


    @torch.no_grad()

    # TODO(myleott): unused, deprecate after pytorch-translate migration

    @torch.no_grad()







class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""






    @torch.jit.export

    @torch.jit.export

    @torch.jit.export

    @torch.jit.export


class SequenceGeneratorWithAlignment(SequenceGenerator):

    @torch.no_grad()



class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

