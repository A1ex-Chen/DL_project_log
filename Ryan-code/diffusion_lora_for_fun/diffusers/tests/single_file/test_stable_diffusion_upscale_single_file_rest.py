import gc
import unittest

import torch

from diffusers import (
    StableDiffusionUpscalePipeline,
)
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
)

from .single_file_testing_utils import SDSingleFileTesterMixin


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionUpscalePipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionUpscalePipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/blob/main/x4-upscaler-ema.safetensors"
    original_config = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/x4-upscaling.yaml"
    repo_id = "stabilityai/stable-diffusion-x4-upscaler"


