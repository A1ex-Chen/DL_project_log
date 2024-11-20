import torch
import torch.nn.functional as F

from .test_base import TestBase
from model.llava.model.builder import load_pretrained_model, get_conv
from model.llava.model.language_model.llava_llama import LlamaForCausalLM
from model.llava.mm_utils import (
    get_model_name_from_path, 
    tokenizer_image_token,
    KeywordsStoppingCriteria
)
from model.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
)
from model.llava.conversation import SeparatorStyle



class TestLLaVA15(TestBase):
    
    

    @torch.no_grad()
            
    @torch.no_grad()