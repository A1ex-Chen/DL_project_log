import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import numpy as np
from .otter.modeling_otter import OtterForConditionalGeneration
from .instruct_blip.models.eva_vit import convert_weights_to_fp16
from .utils import Conversation, SeparatorStyle
from .test_base import TestBase


CONV_VISION = Conversation(
    system='',
    roles=("<image>User", "GPT"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep=" ",
)

class TestOtter(TestBase):


    
    

    