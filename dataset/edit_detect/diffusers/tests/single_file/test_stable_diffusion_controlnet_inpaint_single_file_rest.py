import gc
import tempfile
import unittest

import torch

from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
)

from .single_file_testing_utils import (
    SDSingleFileTesterMixin,
    download_diffusers_config,
    download_original_config,
    download_single_file_checkpoint,
)


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionControlNetInpaintPipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionControlNetInpaintPipeline
    ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/sd-v1-5-inpainting.ckpt"
    original_config = "https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inpainting-inference.yaml"
    repo_id = "runwayml/stable-diffusion-inpainting"









