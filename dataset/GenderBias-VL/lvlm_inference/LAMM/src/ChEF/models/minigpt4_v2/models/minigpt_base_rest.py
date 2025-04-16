import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from ..common.registry import registry
from .base_model import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList

from ..conversation.conversation import StoppingCriteriaSub

class MiniGPTBase(BaseModel):
    """
    Base class for MiniGPT-4 and MiniGPT-v2
    """










    @torch.no_grad()

    @torch.no_grad()