import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.models.llama.modeling_llama import LlamaForCausalLM as LlamaForCausalLMOrig


class LlamaForCausalLM(LlamaForCausalLMOrig):

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)