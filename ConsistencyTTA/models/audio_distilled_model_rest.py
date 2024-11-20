from copy import deepcopy
from typing import Any, Mapping

import torch
import torch.nn as nn

from transformers import CLIPTokenizer, AutoTokenizer
from transformers import CLIPTextModel, T5EncoderModel, AutoModel
from accelerate.logging import get_logger

from diffusers import DDPMScheduler, UNet2DConditionModel, UNet2DConditionGuidedModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from tools.train_utils import do_ema_update

logger = get_logger(__name__, log_level="INFO")


class AudioDistilledModel(nn.Module):


    @property

    @property














    @torch.no_grad()