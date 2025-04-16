from typing import Optional

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import (
    deprecate,
)


class EDICTPipeline(DiffusionPipeline):







    @torch.no_grad()

    @torch.no_grad()

    @torch.no_grad()