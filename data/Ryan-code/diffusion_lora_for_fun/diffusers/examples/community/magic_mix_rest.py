from typing import Union

import torch
from PIL import Image
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)


class MagicMixPipeline(DiffusionPipeline):

    # convert PIL image to latents

    # convert latents to PIL image

    # convert prompt into text embeddings, also unconditional embeddings
