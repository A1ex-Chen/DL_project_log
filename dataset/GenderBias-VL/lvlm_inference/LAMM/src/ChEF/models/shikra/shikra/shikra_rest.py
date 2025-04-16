from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class ShikraConfig(LlamaConfig):
    model_type = "shikra"


class ShikraLlamaModel(LlamaModel):
    config_class = ShikraConfig





class ShikraLlamaForCausalLM(LlamaForCausalLM):
    config_class = ShikraConfig



