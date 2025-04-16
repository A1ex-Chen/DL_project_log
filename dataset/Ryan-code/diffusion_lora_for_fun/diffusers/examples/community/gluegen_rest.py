import inspect
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor

from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TranslatorBase(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2):
        super().__init__()

        self.dim_in = dim
        self.dim_out = dim_out

        self.net_tok = nn.Sequential(
            nn.Linear(num_tok, int(num_tok * mult)),
            nn.LayerNorm(int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), int(num_tok * mult)),
            nn.LayerNorm(int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), num_tok),
            nn.LayerNorm(num_tok),
        )

        self.net_sen = nn.Sequential(
            nn.Linear(dim, int(dim * mult)),
            nn.LayerNorm(int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), int(dim * mult)),
            nn.LayerNorm(int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), dim_out),
            nn.LayerNorm(dim_out),
        )

    def forward(self, x):
        if self.dim_in == self.dim_out:
            indentity_0 = x
            x = self.net_sen(x)
            x += indentity_0
            x = x.transpose(1, 2)

            indentity_1 = x
            x = self.net_tok(x)
            x += indentity_1
            x = x.transpose(1, 2)
        else:
            x = self.net_sen(x)
            x = x.transpose(1, 2)

            x = self.net_tok(x)
            x = x.transpose(1, 2)
        return x


class TranslatorBaseNoLN(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2):
        super().__init__()

        self.dim_in = dim
        self.dim_out = dim_out

        self.net_tok = nn.Sequential(
            nn.Linear(num_tok, int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), num_tok),
        )

        self.net_sen = nn.Sequential(
            nn.Linear(dim, int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), dim_out),
        )

    def forward(self, x):
        if self.dim_in == self.dim_out:
            indentity_0 = x
            x = self.net_sen(x)
            x += indentity_0
            x = x.transpose(1, 2)

            indentity_1 = x
            x = self.net_tok(x)
            x += indentity_1
            x = x.transpose(1, 2)
        else:
            x = self.net_sen(x)
            x = x.transpose(1, 2)

            x = self.net_tok(x)
            x = x.transpose(1, 2)
        return x


class TranslatorNoLN(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2, depth=5):
        super().__init__()

        self.blocks = nn.ModuleList([TranslatorBase(num_tok, dim, dim, mult=2) for d in range(depth)])
        self.gelu = nn.GELU()

        self.tail = TranslatorBaseNoLN(num_tok, dim, dim_out, mult=2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x) + x
            x = self.gelu(x)

        x = self.tail(x)
        return x







class TranslatorBaseNoLN(nn.Module):



class TranslatorNoLN(nn.Module):



def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class GlueGenStableDiffusionPipeline(DiffusionPipeline, StableDiffusionMixin, LoraLoaderMixin):








    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding

    @property

    @property

    @property

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property

    @property

    @property

    @property

    @torch.no_grad()