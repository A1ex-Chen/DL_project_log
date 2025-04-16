import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from ..common.registry import registry
from .base_model import disabled_train
from .minigpt_base import MiniGPTBase
from .Qformer import BertConfig, BertLMHeadModel


@registry.register_model("minigpt4")
class MiniGPT4(MiniGPTBase):
    """
    MiniGPT-4 model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        "pretrain_llama2": "configs/models/minigpt4_llama2.yaml",
    }


    @classmethod


    @classmethod