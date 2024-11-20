""" timm model adapter

Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
from collections import OrderedDict

import torch.nn as nn

try:
    import timm
    from timm.models.layers import Mlp, to_2tuple
    from timm.models.layers.attention_pool2d import RotAttentionPool2d
    from timm.models.layers.attention_pool2d import (
        AttentionPool2d as AbsAttentionPool2d,
    )
except ImportError as e:
    timm = None

from .utils import freeze_batch_norm_2d


class TimmModel(nn.Module):
    """timm model adapter
    # FIXME this adapter is a work in progress, may change in ways that break weight compat
    """


