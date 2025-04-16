import os
import torch
from .test_llava15 import (
    TestLLaVA15, 
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IMAGE_PATCH_TOKEN, 
    get_conv,
    SeparatorStyle
)
from src.model.llava.model import LlavaLlamaForCausalLM
from peft import PeftModel
from transformers import AutoTokenizer

class TestLLaVARLHF(TestLLaVA15):