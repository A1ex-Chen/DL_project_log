import gc
import tempfile
import unittest

import torch

from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
)

from .single_file_testing_utils import (
    SDSingleFileTesterMixin,
    download_original_config,
    download_single_file_checkpoint,
)


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionPipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionPipeline
    ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"
    original_config = (
        "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    )
    repo_id = "runwayml/stable-diffusion-v1-5"








class StableDiffusion21PipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors"
    original_config = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
    repo_id = "stabilityai/stable-diffusion-2-1"



