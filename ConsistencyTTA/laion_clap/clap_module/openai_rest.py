""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
import warnings
from typing import Union, List

import torch

from .model import build_model_from_openai_state_dict
from .pretrained import get_pretrained_url, list_pretrained_tag_models, download_pretrained

__all__ = ["list_openai_models", "load_openai_model"]





    model.apply(patch_device)
    patch_device(model.encode_audio)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()


        model.apply(patch_float)
        patch_float(model.encode_audio)
        patch_float(model.encode_text)
        model.float()

    model.audio_branch.audio_length = model.audio_cfg.audio_length
    return model