# This script is based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

""" LightLLM LLaMA model, compatible with hf"""
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch import distributed as dist

from safetensors import safe_open
from peft import LoraConfig, TaskType
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.modeling_utils import (PreTrainedModel,
                                         GenerationMixin)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.generation import GenerationConfig

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.llama2.model import Llama2TpPartModel

from lightllm.common.basemodel.layer_weights.transformer_layer_weight import TransformerLayerWeight

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LightLLMLlamaConfig"





class LlamaModel:



    


    


class LlamaLightForCausalLM(GenerationMixin):
    
    main_input_name = "input_ids"



    





    
    
    

    
    

    


    # override


    @staticmethod