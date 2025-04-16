from dataclasses import dataclass, field
from peft import LoraConfig, LoraModel, PeftType
from peft.utils import _get_submodules
from peft.tuners.lora import Embedding as LoraEmbedding
import re
import torch.nn as nn
import warnings
from typing import Optional

from .layer import MoeLoraLayer, MoeLinear


@dataclass
class MoeLoraConfig(LoraConfig):

    num_experts: int = field(
        default=16,
        metadata={'help': 'number of experts in MoE Lora Layer'}) 

    gate_mode: str = field(
        default='top2_gate',
        metadata={'help': 'choice: [top2_gate, dual_gate]'}
    )



class MoeLoraModel(LoraModel):

