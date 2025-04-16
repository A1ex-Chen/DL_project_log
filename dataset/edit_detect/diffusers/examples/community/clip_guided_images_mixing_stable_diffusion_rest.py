# -*- coding: utf-8 -*-
import inspect
from typing import Optional, Union

import numpy as np
import PIL.Image
import torch
from torch.nn import functional as F
from torchvision import transforms
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import PIL_INTERPOLATION
from diffusers.utils.torch_utils import randn_tensor










class CLIPGuidedImagesMixingStableDiffusion(DiffusionPipeline, StableDiffusionMixin):









    @torch.enable_grad()

    @torch.no_grad()