from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from einops import rearrange, repeat
from accelerate.hooks import add_hook_to_module, AlignDevicesHook

from .configuration_otter import OtterConfig

import sys

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

XFORMERS_AVAIL = False
try:
    import xformers.ops as xops
    from xformers_model import CLIPVisionModel, LlamaForCausalLM
    from transformers import LlamaTokenizer

    _xformers_version = importlib_metadata.version("xformers")
    print(f"Successfully imported xformers version {_xformers_version}")
except ImportError:
    from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

    XFORMERS_AVAIL = False
    print(
        "No xformers found. You are recommended to install xformers via `pip install xformers` or `conda install -c xformers xformers`"
    )

# from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptneo": "transformer.h",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
}












class OtterPerceiverBlock(nn.Module):



class OtterPerceiverResampler(nn.Module):



class OtterMaskedCrossAttention(nn.Module):



class OtterGatedCrossAttentionBlock(nn.Module):



class OtterLayer(nn.Module):


    # Used this great idea from this implementation of Otter (https://github.com/dhansmair/otter-mini/)





class OtterLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """









class OtterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OtterConfig
    base_model_prefix = "otter"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OtterPerceiverBlock", "CLIPEncoderLayer", "OtterLayer"]



class OtterModel(OtterPreTrainedModel):
    config_class = OtterConfig













class OtterForConditionalGeneration(OtterPreTrainedModel):
    config_class = OtterConfig











    @torch.no_grad()