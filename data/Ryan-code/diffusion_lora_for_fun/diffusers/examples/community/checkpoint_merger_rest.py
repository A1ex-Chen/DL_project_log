import glob
import os
from typing import Dict, List, Union

import safetensors.torch
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import validate_hf_hub_args

from diffusers import DiffusionPipeline, __version__
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME, ONNX_WEIGHTS_NAME, WEIGHTS_NAME


class CheckpointMergerPipeline(DiffusionPipeline):
    """
    A class that supports merging diffusion models based on the discussion here:
    https://github.com/huggingface/diffusers/issues/877

    Example usage:-

    pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", custom_pipeline="checkpoint_merger.py")

    merged_pipe = pipe.merge(["CompVis/stable-diffusion-v1-4","prompthero/openjourney"], interp = 'inv_sigmoid', alpha = 0.8, force = True)

    merged_pipe.to('cuda')

    prompt = "An astronaut riding a unicycle on Mars"

    results = merged_pipe(prompt)

    ## For more details, see the docstring for the merge method.

    """




    @torch.no_grad()
    @validate_hf_hub_args

    @staticmethod

    # Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    @staticmethod

    # Inverse Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    @staticmethod

    @staticmethod