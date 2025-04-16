import inspect
from copy import deepcopy
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
from tqdm.auto import tqdm

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import logging


try:
    from ligo.segments import segment
    from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
except ImportError:
    raise ImportError("Please install transformers and ligo-segments to use the mixture pipeline")

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import LMSDiscreteScheduler, DiffusionPipeline

        >>> scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, custom_pipeline="mixture_tiling")
        >>> pipeline.to("cuda")

        >>> image = pipeline(
        >>>     prompt=[[
        >>>         "A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        >>>         "A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        >>>         "An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
        >>>     ]],
        >>>     tile_height=640,
        >>>     tile_width=640,
        >>>     tile_row_overlap=0,
        >>>     tile_col_overlap=256,
        >>>     guidance_scale=8,
        >>>     seed=7178915308,
        >>>     num_inference_steps=50,
    >>> )["images"][0]
        ```
"""










class StableDiffusionExtrasMixin:
    """Mixin providing additional convenience method to Stable Diffusion pipelines"""



class StableDiffusionTilingPipeline(DiffusionPipeline, StableDiffusionExtrasMixin):

    class SeedTilesMode(Enum):
        """Modes in which the latents of a particular tile can be re-seeded"""

        FULL = "full"
        EXCLUSIVE = "exclusive"

    @torch.no_grad()
