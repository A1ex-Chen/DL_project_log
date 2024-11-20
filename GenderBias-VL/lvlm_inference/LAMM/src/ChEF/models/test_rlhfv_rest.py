import torch
import torch.nn.functional as F
import numpy as np
from .test_base import TestBase
from .rlhfv import (
    init_muffin, 
    wrap_question_with_default_conv, 
    torch_pad_sequence,
    KeywordsStoppingCriteria,
)

class TestRLHFV(TestBase):

    
    
    