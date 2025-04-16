import torch
import torch.nn.functional as F
import numpy as np
from .test_base import TestBase
from .internlm import (
    InternLMXComposer2ForCausalLM, 
    InternLMXcomposer2Config, 
    InternLMXComposer2Tokenizer
)

class TestInternlmXcomposer(TestBase):

    
    
    