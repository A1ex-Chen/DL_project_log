import gc
import unittest

import torch

from diffusers import (
    DDIMScheduler,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
)

from .single_file_testing_utils import SDXLSingleFileTesterMixin


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionXLImg2ImgPipelineSingleFileSlowTests(unittest.TestCase, SDXLSingleFileTesterMixin):
    pipeline_class = StableDiffusionXLImg2ImgPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
    )






@slow
@require_torch_gpu
class StableDiffusionXLImg2ImgRefinerPipelineSingleFileSlowTests(unittest.TestCase):
    pipeline_class = StableDiffusionXLImg2ImgPipeline
    ckpt_path = (
        "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors"
    )
    repo_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml"
    )
