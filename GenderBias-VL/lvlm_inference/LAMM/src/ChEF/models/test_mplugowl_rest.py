import torch
import torch.nn.functional as F
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from .mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from .mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from .utils import Conversation, SeparatorStyle
from .test_base import TestBase
import numpy as np
prompt_template_multi = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"

CONV_VISION = Conversation(
    system="The following is a conversation between a curious human and AI assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("Human", "AI"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="\n",
)
class TestMplugOwl(TestBase):

    

