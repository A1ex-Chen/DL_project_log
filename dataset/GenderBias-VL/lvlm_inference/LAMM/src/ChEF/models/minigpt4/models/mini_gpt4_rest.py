import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from ..common.registry import registry
from .blip2 import Blip2Base, disabled_train
from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer


@registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4.yaml",
    }






    @classmethod