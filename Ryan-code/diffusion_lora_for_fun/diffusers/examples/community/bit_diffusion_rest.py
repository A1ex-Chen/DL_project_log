from typing import Optional, Tuple, Union

import torch
from einops import rearrange, reduce

from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline, ImagePipelineOutput, UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput


BITS = 8


# convert to bit representations and back taken from https://github.com/lucidrains/bit-diffusion/blob/main/bit_diffusion/bit_diffusion.py




# modified scheduler step functions for clamping the predicted x_0 between -bit_scale and +bit_scale




class BitDiffusion(DiffusionPipeline):

    @torch.no_grad()