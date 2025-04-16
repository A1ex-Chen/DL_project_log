import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from ..common.registry import registry
from .base_model import disabled_train
from .minigpt_base import MiniGPTBase
from .Qformer import BertConfig, BertLMHeadModel


@registry.register_model("minigpt_v2")
class MiniGPTv2(MiniGPTBase):
    """
    MiniGPT-v2 model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/minigpt_v2.yaml",
    }



    @classmethod