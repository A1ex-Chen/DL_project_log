import gc
import tempfile
import unittest

import torch

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from .single_file_testing_utils import (
    SDXLSingleFileTesterMixin,
    download_diffusers_config,
    download_single_file_checkpoint,
)


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionXLControlNetPipelineSingleFileSlowTests(unittest.TestCase, SDXLSingleFileTesterMixin):
    pipeline_class = StableDiffusionXLControlNetPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
    )









