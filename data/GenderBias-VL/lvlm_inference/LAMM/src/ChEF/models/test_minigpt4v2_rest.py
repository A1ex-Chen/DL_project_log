import torch
import torch.nn.functional as F
import numpy as np
from .minigpt4_v2.common.config import Config
from .minigpt4_v2.common.registry import registry
from .minigpt4_v2.conversation.conversation import Chat, CONV_VISION_minigptv2, CONV_VISION_Vicuna0

# imports modules for registration
from .minigpt4_v2.models import *
from .minigpt4_v2.processors import *
from .test_base import TestBase
from .utils import get_image
from .minigpt4_v2.common.eval_utils import prepare_texts, eval_parser

class TestMiniGPT4V2(TestBase):
        # self.chat.move_stopping_criteria_device(self.device, dtype=self.dtype)

    

    @torch.no_grad()
    
    @torch.no_grad()
    
    