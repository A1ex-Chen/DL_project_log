import gc
import unittest

import torch

from diffusers import (
    StableDiffusionInpaintPipeline,
)
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
)

from .single_file_testing_utils import SDSingleFileTesterMixin


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionInpaintPipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionInpaintPipeline
    ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/sd-v1-5-inpainting.ckpt"
    original_config = "https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inpainting-inference.yaml"
    repo_id = "runwayml/stable-diffusion-inpainting"







@slow
@require_torch_gpu
class StableDiffusion21InpaintPipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionInpaintPipeline
    ckpt_path = (
        "https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/512-inpainting-ema.safetensors"
    )
    original_config = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inpainting-inference.yaml"
    repo_id = "stabilityai/stable-diffusion-2-inpainting"



