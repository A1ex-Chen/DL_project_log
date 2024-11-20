import torch
import torch.nn.functional as F

from . import llama_adapter_v2
from .utils import *
from .test_base import TestBase

CONV_VISION = Conversation(
    system='Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n\n',
    roles=("Instruction", "Input", "Response"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="### ",
)
class TestLLamaAdapterV2(TestBase):

    

    
    