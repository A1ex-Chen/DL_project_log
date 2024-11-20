import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import torch
from numpy import exp, pi, sqrt
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler




@dataclass
class CanvasRegion:
    """Class defining a rectangular region in the canvas"""

    row_init: int  # Region starting row in pixel space (included)
    row_end: int  # Region end row in pixel space (not included)
    col_init: int  # Region starting column in pixel space (included)
    col_end: int  # Region end column in pixel space (not included)
    region_seed: int = None  # Seed for random operations in this region
    noise_eps: float = 0.0  # Deviation of a zero-mean gaussian noise to be applied over the latents in this region. Useful for slightly "rerolling" latents


    @property

    @property


    @property


class MaskModes(Enum):
    """Modes in which the influence of diffuser is masked"""

    CONSTANT = "constant"
    GAUSSIAN = "gaussian"
    QUARTIC = "quartic"  # See https://en.wikipedia.org/wiki/Kernel_(statistics)


@dataclass
class DiffusionRegion(CanvasRegion):
    """Abstract class defining a region where some class of diffusion process is acting"""

    pass


@dataclass
class Text2ImageRegion(DiffusionRegion):
    """Class defining a region where a text guided diffusion process is acting"""

    prompt: str = ""  # Text prompt guiding the diffuser in this region
    guidance_scale: float = 7.5  # Guidance scale of the diffuser in this region. If None, randomize
    mask_type: MaskModes = MaskModes.GAUSSIAN.value  # Kind of weight mask applied to this region
    mask_weight: float = 1.0  # Global weights multiplier of the mask
    tokenized_prompt = None  # Tokenized prompt
    encoded_prompt = None  # Encoded prompt





@dataclass
class Image2ImageRegion(DiffusionRegion):
    """Class defining a region where an image guided diffusion process is acting"""

    reference_image: torch.Tensor = None
    strength: float = 0.8  # Strength of the image



    @property


class RerollModes(Enum):
    """Modes in which the reroll regions operate"""

    RESET = "reset"  # Completely reset the random noise in the region
    EPSILON = "epsilon"  # Alter slightly the latents in the region


@dataclass
class RerollRegion(CanvasRegion):
    """Class defining a rectangular canvas region in which initial latent noise will be rerolled"""

    reroll_mode: RerollModes = RerollModes.RESET.value


@dataclass
class MaskWeightsBuilder:
    """Auxiliary class to compute a tensor of weights for a given diffusion region"""

    latent_space_dim: int  # Size of the U-net latent space
    nbatch: int = 1  # Batch size in the U-net






class StableDiffusionCanvasPipeline(DiffusionPipeline, StableDiffusionMixin):
    """Stable Diffusion pipeline that mixes several diffusers in the same canvas"""




    @torch.no_grad()