import torch
import torch.nn.functional as F
import numpy as np
from .minigpt4.common.config import Config
from .minigpt4.common.registry import registry
from .minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from .minigpt4.models import *
from .minigpt4.processors import *
from .test_base import TestBase
from .utils import get_image
class TestMiniGPT4(TestBase):

    

    @torch.no_grad()
    
    
    