import gc
import unittest

import torch

from diffusers import StableDiffusionXLInstructPix2PixPipeline
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
)


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionXLInstructPix2PixPipeline(unittest.TestCase):
    pipeline_class = StableDiffusionXLInstructPix2PixPipeline
    ckpt_path = "https://huggingface.co/stabilityai/cosxl/blob/main/cosxl_edit.safetensors"
    original_config = None
    repo_id = "diffusers/sdxl-instructpix2pix-768"



