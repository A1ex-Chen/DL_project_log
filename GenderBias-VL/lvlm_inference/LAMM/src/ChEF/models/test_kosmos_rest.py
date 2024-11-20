import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from collections import namedtuple
import sentencepiece as spm
import ast
from fairseq_cli.generate import get_symbols_to_strip_from_output

import torch.nn.functional as F
from .kosmos2.utils import get_interactive_tokens_and_lengths, post_process_prediction, get_token_src
from .test_base import TestBase
from .kosmos2 import unilm

Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints img_src_tokens img_gpt_input_mask")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")

class TestKOSMOS2(TestBase): # TODO: batch_size = 1




    

    