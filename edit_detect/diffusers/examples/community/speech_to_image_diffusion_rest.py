import inspect
from typing import Callable, List, Optional, Union

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.pipeline_utils import StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SpeechToImagePipeline(DiffusionPipeline, StableDiffusionMixin):

    @torch.no_grad()