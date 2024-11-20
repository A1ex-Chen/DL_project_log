from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


class MaskedStableDiffusionImg2ImgPipeline(StableDiffusionImg2ImgPipeline):
    debug_save = False

    @torch.no_grad()
