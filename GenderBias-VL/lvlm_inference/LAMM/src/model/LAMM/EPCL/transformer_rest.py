# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified from DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
from typing import Optional

import torch
from torch import Tensor, nn

from .helpers import ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT, get_clones


class TransformerEncoder(nn.Module):




class TransformerDecoder(nn.Module):




class MaskedTransformerEncoder(TransformerEncoder):





class TransformerEncoderLayer(nn.Module):







class TransformerDecoderLayer(nn.Module):



